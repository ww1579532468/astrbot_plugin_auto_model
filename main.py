import asyncio
import re
import random
import json
from urllib.parse import urlsplit
from types import SimpleNamespace

import aiohttp
from aiohttp import web
from astrbot.api import AstrBotConfig
from astrbot.api.event import filter
from astrbot.api.star import Context, Star

# 本机接口和资源限制。
MODEL_NAME = "auto"  # 自动注册的供应商标识和默认模型别名。
LOCAL_BASE = "http://127.0.0.1/v1"  # 注册给 AstrBot 的本机接口地址；监听地址和端口在 initialize 中设置。
DEFAULT_CONTEXT = 100000  # 默认上下文长度参考值；当前各处默认值仍单独写为 100000，修改时需同步。
MAX_RESPONSE_BYTES = 32 * 1024 * 1024  # 单次非流式上游响应的大小上限，单位字节，当前为 32 MiB。
MAX_MODELS_BYTES = 4 * 1024 * 1024  # 上游模型列表的大小上限，单位字节，当前为 4 MiB。
MODELS_TIMEOUT = 15  # 获取上游模型列表的总超时秒数；不用于会话请求。

# 自动兜底选择规则；配置中明确指定的优先模型不受这些筛选限制。
AUTO_PREFERENCE = ("flash", "low")  # 自动选择顺序：先找 flash，再找 low，都没有则选最后一个合格模型。
ALLOWED_EFFORT = {"low", "minimal", "none", "off"}  # 思考模型名称包含这些强度标记时，允许参与自动选择。
BLOCKED_EFFORT = {"high", "xhigh", "ultra", "max", "heavy", "extended"}  # 高消耗强度黑名单，命中任意一项即排除。
IMAGE_MARKERS = ("image", "imagen", "dall-e", "dalle", "stable-diffusion", "flux", "midjourney", "ideogram", "recraft")  # 绘图模型名称黑名单，命中任意一项即排除。
NON_CHAT_MARKERS = {"embedding", "embeddings", "rerank", "reranker", "whisper", "tts", "moderation", "sora", "veo", "video", "transcribe", "transcription"}  # 非对话模型关键词黑名单，命中任意一项即排除。


# 将配置错误与其他异常区分，便于返回接口错误信息。
class ConfigError(ValueError):
    pass


# 统一地址格式，用于判断已有供应商是否指向本机接口。
def normalized_url(value):
    url = urlsplit(value)
    return (
        url.scheme.lower(),
        url.hostname,
        url.port or (443 if url.scheme == "https" else 80),
        url.path.rstrip("/"),
        url.query,
        url.fragment,
    )


# 从模型名称提取数字版本，供自动选择时比较新旧。
def model_version(name):
    # 忽略较长的日期编号和渠道路径前缀。
    name = name.lower().rsplit("/", 1)[-1]
    versions = re.findall(r"(?<![a-z0-9])v?(\d{1,3}(?:\.\d+)*)(?![a-z0-9])", name)
    return tuple(int(n) for n in versions[0].split(".")) if versions else ()


# 将名称拆成小写词段，避免把单词内部的字符误判为强度标记。
def name_parts(name):
    return set(re.findall(r"[a-z0-9]+", name.lower()))


# 筛除绘图、非对话和高消耗思考模型。
def eligible_auto_model(name):
    parts = name_parts(name)
    if parts & (BLOCKED_EFFORT | NON_CHAT_MARKERS):
        return False
    if any(
        re.search(r"(?<![a-z0-9])" + re.escape(marker) + r"(?![a-z0-9])", name.lower())
        for marker in IMAGE_MARKERS
    ):
        return False
    # 未标注允许强度的思考模型可能消耗大量思考令牌。
    if parts & {"thinking", "reasoning"} and not parts & ALLOWED_EFFORT:
        return False
    return True


class AutoModelPlugin(Star):
    # 保存配置并初始化并发锁、HTTP 客户端和请求任务状态。
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        # 界面中的四项配置：当前模型会持久保存，其他项由用户手动填写。
        # 当前模型：实际转发使用的上游模型名称，仅初始化选择或失败后刷新时更新。
        # API：上游接口基础地址，通常以 /v1 结尾。
        # key：访问上游所需的密钥，本机接口不校验密钥。
        # 优先级：每行填写 模型名称|上下文长度，按从上到下的顺序精确匹配。
        self.config = config
        self.lock = asyncio.Lock()
        self.runner = None
        self.client = None
        self.registration_lock = asyncio.Lock()
        self.selection_task = None
        self.active_requests = set()
        self.registration_error = None

    # 读取并校验文本配置，兼容未填写的情况。
    def text(self, key):
        value = self.config.get(key)
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ConfigError(f"Configuration field {key} must be text")
        return value.strip()

    # 将优先级配置统一转换为逐行文本。
    def priority_lines(self):
        lines = self.config.get("优先级")
        if lines is None:
            return []
        if isinstance(lines, str):
            return lines.splitlines()
        if isinstance(lines, list) and all(isinstance(line, str) for line in lines):
            return lines
        raise ConfigError("Priority must be text or a list of text lines")

    # 解析模型名和上下文长度，省略长度时使用 100000。
    def priorities(self):
        lines = self.priority_lines()
        result = []
        for number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            name, sep, size = line.strip().rpartition("|")
            if not sep:
                name, size = line.strip(), "100000"
            name = name.strip()
            if not name:
                raise ConfigError(f"Priority line {number}: model name must not be empty")
            try:
                tokens = int(size)
                if tokens <= 0:
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    f"Priority line {number}: context must be a positive integer"
                ) from None
            result.append((name, tokens))
        return result

    # 校验上游基础地址，并阻止指向本机路由的循环请求。
    def upstream(self):
        base = self.text("API").rstrip("/")
        if not base:
            raise ConfigError("Configure upstream API first")
        # 避免上游地址指向本服务，形成循环转发。
        try:
            url = urlsplit(base)
            port = url.port
        except ValueError:
            raise ConfigError("Invalid upstream API URL or port") from None
        if port is not None and port < 1:
            raise ConfigError("Upstream API port must be between 1 and 65535")
        if url.scheme not in ("http", "https") or not url.hostname:
            raise ConfigError("Invalid upstream API URL")
        if url.username or url.password or url.query or url.fragment:
            raise ConfigError("API URL must not contain credentials, query or fragment")
        if (
            url.hostname in ("localhost", "0.0.0.0", "::", "::1")
            or url.hostname.startswith("127.")
        ) and (port or (80 if url.scheme == "http" else 443)) == 80:
            raise ConfigError("Upstream cannot be this local service")
        return base

    # 使用配置的上游密钥生成鉴权请求头。
    def headers(self):
        key = self.text("key")
        return {"Authorization": "Bearer " + key} if key else {}

    # 从 flash 候选中选择数字版本最高的模型，同版本优先名称以 flash 结尾的模型。
    @staticmethod
    def latest_flash(available):
        def rank(name):
            lowered = name.lower()
            # 识别版本时，忽略 flash 后面的日期和后缀数字。
            prefix = lowered.split("flash", 1)[0]
            version = model_version(prefix)
            return version, lowered.endswith("flash"), lowered

        return max((n for n in available if "flash" in n.lower()), key=rank, default="")

    # 手动优先级没有匹配时，按顶部规则自动兜底选择。
    @classmethod
    def automatic_model(cls, available):
        candidates = [name for name in available if eligible_auto_model(name)]
        for preference in AUTO_PREFERENCE:
            if preference == "flash":
                chosen = cls.latest_flash(candidates)
            else:
                matching = [
                    name for name in candidates if preference in name_parts(name)
                ]

                def version(name):
                    value = model_version(name)
                    return value, name.lower()

                chosen = max(matching, key=version, default="")
            if chosen:
                return chosen
        return candidates[-1] if candidates else ""

    # 合并并发刷新，让同一批失败请求共享一次模型列表查询。
    async def select(self, failed):
        if self.selection_task is None or self.selection_task.done():
            self.selection_task = asyncio.create_task(self._select(failed))
        return await asyncio.shield(self.selection_task)

    # 分块读取上游数据并限制大小，避免响应占用过多内存。
    @staticmethod
    async def limited_read(response, limit):
        data = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            if len(data) + len(chunk) > limit:
                raise ValueError("Upstream response exceeds size limit")
            data.extend(chunk)
        return bytes(data)

    # 查询可用模型、按优先级选择并保存；没有新结果时保留当前模型。
    async def _select(self, failed):
        async with self.lock:
            current = self.text("当前模型")
            if current != failed:
                return current
            async with self.client.get(
                self.upstream() + "/models",
                headers=self.headers(),
                allow_redirects=False,
                timeout=aiohttp.ClientTimeout(total=MODELS_TIMEOUT),
            ) as response:
                response.raise_for_status()
                data = json.loads(await self.limited_read(response, MAX_MODELS_BYTES))
            # 等待查询期间用户可能手动更改当前模型，保留最新设置。
            latest = self.text("当前模型")
            if latest != current:
                return latest
            if not isinstance(data, dict) or not isinstance(data.get("data"), list):
                raise ValueError("Invalid upstream model list")
            available = list(
                dict.fromkeys(
                    item["id"]
                    for item in data["data"]
                    if isinstance(item, dict)
                    and isinstance(item.get("id"), str)
                    and item["id"].strip()
                )
            )
            candidate = next(
                (name for name, _ in self.priorities() if name in available), ""
            )
            updates = {}
            if not candidate:
                candidate = self.automatic_model(available)
                if candidate:
                    lines = "\n".join(self.priority_lines())
                    updates["优先级"] = (
                        lines.rstrip()
                        + ("\n" if lines.strip() else "")
                        + candidate
                    )
            if candidate:
                if candidate != current:
                    updates["当前模型"] = candidate
            if updates:
                previous = {key: self.config.get(key, "") for key in updates}
                try:
                    self.config.save_config(updates)
                except Exception:
                    self.config.update(previous)
                    raise
            if candidate and candidate != current:
                self.logger.info(
                    "Model switched: %s -> %s", current or "(empty)", candidate
                )
                return candidate
            return current

    # 读取 AstrBot 已注册的本机模型名称，用于模型列表展示。
    def public_model(self):
        manager = getattr(self.context, "provider_manager", None)
        config = getattr(getattr(manager, "acm", None), "default_conf", {})
        if isinstance(config, dict):
            for item in config.get("provider", []):
                if item.get("provider_source_id") == MODEL_NAME and item.get(
                    "model"
                ) in ("auto", "otto"):
                    return item["model"]
        return MODEL_NAME

    # 处理本机 GET /v1/models 请求。
    async def models(self, request):
        try:
            return self.model_listing()
        except ConfigError as exc:
            return self.error(str(exc), 503)

    # 生成本机模型列表，附带当前上下文长度和注册失败状态。
    def model_listing(self):
        current = self.text("当前模型")
        size = next(
            (size for name, size in self.priorities() if name == current), 100000
        )
        names = list(dict.fromkeys(("auto", self.public_model())))
        result = {
            "object": "list",
            "data": [
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "auto",
                    "context_window": size,
                }
                for name in names
            ],
        }
        if self.registration_error:
            result["registration_error"] = self.registration_error
        return web.json_response(result)

    # 统一生成兼容 OpenAI 格式的错误响应。
    @staticmethod
    def error(message, status=502):
        return web.json_response(
            {"error": {"message": message, "type": "api_error", "code": None}},
            status=status,
        )

    # 管理会话转发任务；客户端断开时取消对应上游请求。
    async def responses(self, request):
        task = asyncio.create_task(self.forward(request))
        self.active_requests.add(task)
        transport = request.transport

        async def disconnected():
            while transport is not None and not transport.is_closing():
                await asyncio.sleep(0.25)

        watcher = asyncio.create_task(disconnected()) if transport is not None else None
        try:
            if watcher:
                done, _ = await asyncio.wait(
                    (task, watcher), return_when=asyncio.FIRST_COMPLETED
                )
                if task not in done:
                    task.cancel()
                    raise ConnectionResetError("Client disconnected")
            return await task
        finally:
            task.cancel()
            if watcher:
                watcher.cancel()
            await asyncio.gather(
                task, *([watcher] if watcher else []), return_exceptions=True
            )
            self.active_requests.discard(task)
            if (
                not self.active_requests
                and self.selection_task
                and not self.selection_task.done()
            ):
                self.selection_task.cancel()
                await asyncio.gather(self.selection_task, return_exceptions=True)

    # 校验会话请求并转发；失败时刷新模型，仅模型改变后重试一次。
    async def forward(self, request):
        try:
            body = await request.json()
        except web.HTTPRequestEntityTooLarge:
            return self.error("Request exceeds size limit", 413)
        except Exception:
            return self.error("Invalid JSON body", 400)
        if not isinstance(body, dict):
            return self.error("Request body must be a JSON object", 400)
        if not isinstance(body.get("model"), str) or not body["model"].strip():
            return self.error("model must be a non-empty string", 400)
        # 请求中的模型名仅作为别名；实际转发使用配置中缓存的当前模型。
        if "stream" in body and not isinstance(body["stream"], bool):
            return self.error("stream must be boolean", 400)
        try:
            current = self.text("当前模型")
            if not current:
                current = await self.select("")
            if not current:
                return self.error("No configured model is available", 503)
            for attempt in range(2):
                payload = dict(body, model=current)
                try:
                    response = await self.client.post(
                        self.upstream() + "/responses",
                        json=payload,
                        headers=self.headers(),
                        allow_redirects=False,
                    )
                except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                    response = None
                if response is None or response.status >= 300:
                    if response is not None:
                        # 上游错误可能含有私密信息，不直接返回原始内容。
                        status = response.status
                        response.release()
                    else:
                        status = 502
                    if attempt == 0:
                        try:
                            selected = await self.select(current)
                        except Exception:
                            selected = current
                        if selected and selected != current:
                            current = selected
                            continue
                    return self.error("Upstream request failed", status)
                # 按请求的 stream 参数转发流式或非流式响应。
                async with response:
                    if body.get("stream", False):
                        downstream = web.StreamResponse(
                            status=response.status,
                            headers={
                                "Content-Type": "text/event-stream",
                                "Cache-Control": "no-cache",
                                "X-Accel-Buffering": "no",
                            },
                        )
                        await downstream.prepare(request)
                        try:
                            async for chunk in response.content.iter_any():
                                await downstream.write(chunk)
                            await downstream.write_eof()
                        except (aiohttp.ClientError, ConnectionError, OSError):
                            # 已发送流式响应头或内容后，不再重试请求。
                            downstream.force_close()
                            request.transport.close() if request.transport else None
                        return downstream
                    try:
                        data = await self.limited_read(response, MAX_RESPONSE_BYTES)
                    except ValueError:
                        return self.error("Upstream response exceeds size limit", 502)
                    except (aiohttp.ClientError, OSError):
                        if attempt == 0:
                            try:
                                selected = await self.select(current)
                            except Exception:
                                selected = current
                            if selected and selected != current:
                                current = selected
                                continue
                        return self.error("Upstream response interrupted")
                    headers = {
                        name: response.headers[name]
                        for name in ("Content-Type", "X-Request-Id", "Retry-After")
                        if name in response.headers
                    }
                    return web.Response(
                        body=data, status=response.status, headers=headers
                    )
        except ConfigError as exc:
            return self.error(str(exc), 503)
        except Exception:
            return self.error("Auto service configuration or upstream error")

    # 启动仅监听本机的 HTTP 服务，并在条件满足时注册 AstrBot 模型。
    async def initialize(self):
        try:
            self.priorities()
            self.client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=None), trust_env=False
            )
            app = web.Application(client_max_size=32 * 1024 * 1024)
            app.router.add_get("/v1/models", self.models)
            app.router.add_post("/v1/responses", self.responses)
            self.runner = web.AppRunner(app, access_log=None)
            await self.runner.setup()
            await web.TCPSite(self.runner, "127.0.0.1", 80).start()
            if getattr(
                getattr(self.context, "provider_manager", None), "provider_insts", None
            ):
                await self.ensure_model()
        except Exception:
            await self.terminate()
            raise

    # AstrBot 加载完成后确认模型已注册，失败状态通过模型列表接口提供。
    @filter.on_astrbot_loaded()
    async def ensure_model(self):
        try:
            await self._ensure_model()
            self.registration_error = None
        except Exception:
            self.registration_error = "Automatic model registration failed; check AstrBot version and local provider configuration"

    # 使用 AstrBot 官方服务和管理器创建或补齐供应商、模型及界面字段。
    async def _ensure_model(self):
        from astrbot.dashboard.services.config_service import ProviderConfigService

        async with self.registration_lock:
            manager = self.context.provider_manager
            config = manager.acm.default_conf
            source_id = MODEL_NAME
            existing_model = next(
                (
                    p
                    for p in config.get("provider", [])
                    if p.get("provider_source_id") == source_id
                    and p.get("model") in ("auto", "otto")
                ),
                None,
            )
            chosen_name = (
                existing_model["model"]
                if existing_model
                else (
                    "otto" if random.random() < 0.01 else MODEL_NAME
                )
            )
            model_id = (
                existing_model["id"] if existing_model else f"{source_id}/{chosen_name}"
            )
            if existing_model is None and any(
                p.get("id") == model_id for p in config.get("provider", [])
            ):
                raise RuntimeError("Model ID already belongs to a different model")
            service = ProviderConfigService(
                SimpleNamespace(astrbot_config=config, provider_manager=manager)
            )
            sources = config.get("provider_sources", [])
            source = next((s for s in sources if s.get("id") == source_id), None)
            # 没有供应商时创建；已有供应商则校验地址并补齐缺失字段。
            if source is None:
                await service.upsert_provider_source(
                    source_id,
                    {
                        "id": source_id,
                        "type": "openai_responses",
                        "provider": "openai",
                        "provider_type": "chat_completion",
                        "api_base": LOCAL_BASE,
                        "key": ["auto"],
                        "enable": True,
                        "timeout": 120,
                        "proxy": "",
                        "custom_headers": {},
                    },
                )
            elif source.get("type") != "openai_responses" or normalized_url(
                source.get("api_base") or ""
            ) != normalized_url(LOCAL_BASE):
                raise RuntimeError(
                    "Provider source auto already exists with different settings"
                )
            elif "proxy" not in source or "custom_headers" not in source:
                updated_source = dict(source)
                updated_source.setdefault("proxy", "")
                updated_source.setdefault("custom_headers", {})
                await service.upsert_provider_source(source_id, updated_source)
            # 已有模型只补齐缺失项，并整理启用、能力、参数、上下文的显示顺序。
            if existing_model is not None:
                updated_model = dict(existing_model)
                updated_model.setdefault("modalities", ["text", "image", "tool_use"])
                updated_model.setdefault("custom_extra_body", {})
                ordered_fields = (
                    "enable",
                    "modalities",
                    "custom_extra_body",
                    "max_context_tokens",
                )
                ordered = {
                    k: v for k, v in updated_model.items() if k not in ordered_fields
                }
                ordered.update(
                    {k: updated_model[k] for k in ordered_fields if k in updated_model}
                )
                updated_model = ordered
                if updated_model != existing_model or list(updated_model) != list(
                    existing_model
                ):
                    await manager.update_provider(model_id, updated_model)
                return
            # 没有已注册的本机模型时，创建带默认能力和空请求体参数的模型。
            await manager.create_provider(
                {
                    "id": model_id,
                    "provider_source_id": source_id,
                    "model": chosen_name,
                    "provider_type": "chat_completion",
                    "enable": True,
                    "modalities": ["text", "image", "tool_use"],
                    "custom_extra_body": {},
                    "max_context_tokens": 100000,
                }
            )

    # 插件卸载时取消后台任务，并关闭 HTTP 服务和上游客户端。
    async def terminate(self):
        try:
            tasks = list(self.active_requests)
            if self.selection_task:
                tasks.append(self.selection_task)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.runner:
                await self.runner.cleanup()
        finally:
            self.runner = None
            if self.client:
                await self.client.close()
                self.client = None
