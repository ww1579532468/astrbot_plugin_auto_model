import asyncio
import ipaddress
import json
import random
import re
import secrets
import time
from types import SimpleNamespace
from urllib.parse import urlsplit

import aiohttp
from aiohttp import web

from astrbot.api import AstrBotConfig
from astrbot.api.event import filter
from astrbot.api.star import Context, Star

# 常量：接口默认值、刷新间隔和响应大小限制。
MODEL_NAME = "auto"  # 自动注册的供应商标识和默认模型别名。
DEFAULT_PORT = 56751  # HTTP 默认监听端口，可在配置界面修改。
DEFAULT_CONTEXT = 100000  # 未指定模型上下文长度时使用的默认值。
MAX_RESPONSE_BYTES = 32 * 1024 * 1024  # 非流式响应大小上限，单位字节。
MAX_MODELS_BYTES = 4 * 1024 * 1024  # 上游模型列表大小上限，单位字节。
MAX_ERROR_BYTES = 64 * 1024  # 只读取有限错误内容用于识别模型不可用，不输出原文。
MODELS_TIMEOUT = 15  # 模型列表查询超时秒数，不用于会话请求。
MODELS_INTERVAL = 1  # 一秒内的刷新共用结果，进行中的查询始终共用。

# 自动兜底选择规则；配置中明确指定的优先模型不受这些筛选限制。
AUTO_PREFERENCE = (
    "flash",
    "low",
)  # 自动选择顺序：先找 flash，再找 low，都没有则选最后一个合格模型。
ALLOWED_EFFORT = {
    "low",
    "minimal",
    "none",
    "off",
}  # 思考模型名称包含这些强度标记时，允许参与自动选择。
BLOCKED_EFFORT = {
    "high",
    "xhigh",
    "ultra",
    "max",
    "heavy",
    "extended",
}  # 高消耗强度黑名单，命中任意一项即排除。
IMAGE_MARKERS = (
    "image",
    "imagen",
    "dall-e",
    "dalle",
    "stable-diffusion",
    "flux",
    "midjourney",
    "ideogram",
    "recraft",
)  # 绘图模型名称黑名单，命中任意一项即排除。
NON_CHAT_MARKERS = {
    "embedding",
    "embeddings",
    "rerank",
    "reranker",
    "whisper",
    "tts",
    "moderation",
    "sora",
    "veo",
    "video",
    "transcribe",
    "transcription",
    "audio",
    "realtime",
    "speech",
    "asr",
}  # 非对话模型关键词黑名单，命中任意一项即排除。


# 将配置错误与其他异常区分，便于返回接口错误信息。
class ConfigError(ValueError):
    pass


# 从模型名称提取数字版本，供自动选择时比较新旧。
def model_version(name):
    # 忽略较长的日期编号和渠道路径前缀。
    name = name.lower().rsplit("/", 1)[-1]
    versions = re.findall(r"(?<![0-9.])(\d{1,3}(?:\.\d+)*)(?![a-z0-9.])", name)
    version = [int(n) for n in versions[0].split(".")] if versions else []
    # 3 和 3.0 视为同一版本；长日期不截取，小数版本不从中间重新匹配。
    while len(version) > 1 and version[-1] == 0:
        version.pop()
    return tuple(version)


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
        # 当前模型：实际自动路由使用的上游名称，模型检查发现缺失时更新并持久保存。
        # API：上游接口基础地址，通常以 /v1 结尾。
        # key：访问上游所需的密钥，与对外访问密钥无关。
        # 优先级：每行填写 模型名称|上下文长度，按从上到下的顺序精确匹配。
        # listen_port：HTTP 接口端口，默认 56751。
        # 允许外部访问：关闭时绑定本机，开启时绑定全部 IPv4 网卡。
        # 访问密钥：外部请求的 Bearer 密钥；开启外部访问且留空时生成并保存。
        self.config = config
        self.port = config.get("listen_port", DEFAULT_PORT)
        if (
            isinstance(self.port, bool)
            or not isinstance(self.port, int)
            or not 1 <= self.port <= 65535
        ):
            raise ConfigError("listen_port must be an integer between 1 and 65535")
        self.local_base = f"http://127.0.0.1:{self.port}/v1"
        self.lock = asyncio.Lock()
        self.runner = None
        self.client = None
        self.registration_lock = asyncio.Lock()
        self.selection_task = None
        self.active_requests = set()
        self.registration_error = None
        self.available_models = None
        self.last_models_attempt = None
        self.missing_models = set()
        self.models_generation = 0
        self.allow_external = config.get("允许外部访问", False)
        if not isinstance(self.allow_external, bool):
            raise ConfigError("允许外部访问 must be boolean")
        self.access_key = self.text("访问密钥")
        if self.allow_external and not self.access_key:
            self.access_key = secrets.token_urlsafe(32)
            self.config.save_config({"访问密钥": self.access_key})

    # 配置处理：读取并校验文本配置，兼容未填写的情况。
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
                name, size = line.strip(), str(DEFAULT_CONTEXT)
            name = name.strip()
            if not name:
                raise ConfigError(
                    f"Priority line {number}: model name must not be empty"
                )
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
        ) and (port or (80 if url.scheme == "http" else 443)) == self.port:
            raise ConfigError("Upstream cannot be this local service")
        return base

    # 使用配置的上游密钥生成鉴权请求头。
    def headers(self):
        key = self.text("key")
        return {"Authorization": "Bearer " + key} if key else {}

    # 上游请求：获取列表、发送会话和受限读取，不负责路由决策。
    async def fetch_models(self):
        async with self.client.get(
            self.upstream() + "/models",
            headers=self.headers(),
            allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=MODELS_TIMEOUT),
        ) as response:
            response.raise_for_status()
            data = json.loads(await self.limited_read(response, MAX_MODELS_BYTES))
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("Invalid upstream model list")
        return list(
            dict.fromkeys(
                item["id"]
                for item in data["data"]
                if isinstance(item, dict)
                and isinstance(item.get("id"), str)
                and item["id"].strip()
            )
        )

    async def request_response(self, payload):
        return await self.client.post(
            self.upstream() + "/responses",
            json=payload,
            headers=self.headers(),
            allow_redirects=False,
        )

    @staticmethod
    async def limited_read(response, limit):
        data = bytearray()
        async for chunk in response.content.iter_chunked(65536):
            if len(data) + len(chunk) > limit:
                raise ValueError("Upstream response exceeds size limit")
            data.extend(chunk)
        return bytes(data)

    # 仅识别明确的模型错误，避免将鉴权、限流及普通服务故障记录为模型不存在。
    @staticmethod
    def model_unavailable(status, data):
        if status not in (400, 403, 404, 422, 500, 503):
            return False
        try:
            value = json.loads(data)
        except (ValueError, UnicodeError):
            return False
        if not isinstance(value, dict) or not isinstance(value.get("error"), dict):
            return False
        error = value["error"]
        # 错误明确指向工具、输入等其他参数时，不把请求模型记为不存在。
        if error.get("param") not in (None, "", "model"):
            return False
        codes = {str(error.get(key, "")).lower() for key in ("code", "type")}
        if codes & {
            "model_not_found",
            "invalid_model",
            "model_not_available",
            "model_unavailable",
            "no_available_channel",
        }:
            return True
        message = str(error.get("message", "")).lower()
        # 缺少名为 model 的工具参数也可能含有 not found，消息兜底要保守处理。
        if re.search(
            r"\b(?:tools?|functions?|arguments?|parameters?|fields?|properties|schema)\b"
            r"|工具|函数|参数|字段",
            message,
        ):
            return False
        english = bool(re.search(r"\bmodel\b", message)) and any(
            marker in message
            for marker in (
                "does not exist",
                "not found",
                "not available",
                "unavailable",
                "no available channel",
            )
        )
        chinese = "模型" in message and any(
            marker in message
            for marker in ("不存在", "不可用", "无可用渠道", "没有可用渠道", "未找到")
        )
        return english or chinese

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

    # 缓存与模型选择：刷新仅在请求触发时执行，没有定时轮询。
    async def select(self, failed):
        current = self.text("当前模型")
        if current != failed:
            return current
        await self.refresh_models()
        return self.text("当前模型")

    # 共用正在进行或一秒内已结束的查询；取消等待者不影响其他调用。
    async def refresh_models(self, missing=None):
        # missing 为本次明确失败的名称及请求代次；过期请求不能污染新缓存。
        remember = missing is not None and missing[1] == self.models_generation
        now = time.monotonic()
        if self.selection_task is None or (
            self.selection_task.done()
            and (
                self.last_models_attempt is None
                or now - self.last_models_attempt >= MODELS_INTERVAL
            )
        ):
            self.last_models_attempt = now
            self.selection_task = asyncio.create_task(self._refresh_models())
            self.selection_task.add_done_callback(
                lambda task: task.exception() if not task.cancelled() else None
            )
        generation = await asyncio.shield(self.selection_task)
        # 本次回退主动刷新的列表仍无此名称，才保留本次不存在记录。
        # 后续成功刷新照常清除；新列表已含此名称时允许下次重新尝试。
        if (
            remember
            and generation == self.models_generation
            and missing[0] not in (self.available_models or [])
        ):
            self.missing_models.add(missing[0])
        return self.text("当前模型")

    # 查询成功才替换列表并清除不存在记录；查询期间的手动模型修改优先。
    async def _refresh_models(self):
        async with self.lock:
            current = self.text("当前模型")
            available = await self.fetch_models()
            self.available_models = available
            self.missing_models.clear()
            self.models_generation += 1
            latest = self.text("当前模型")
            if latest == current and current not in available:
                candidate, append = self.choose_model(available)
                self.save_selection(current, candidate, append)
            return self.models_generation

    # 候选选择只计算结果，明确配置的优先级不受自动筛选规则限制。
    def choose_model(self, available):
        candidate = next(
            (name for name, _ in self.priorities() if name in available), ""
        )
        if candidate:
            return candidate, False
        return self.automatic_model(available), True

    # 持久保存切换；保存失败恢复内存配置，自动补选只追加模型名。
    def save_selection(self, current, candidate, append):
        updates = {}
        if (
            candidate
            and append
            and candidate not in {name for name, _ in self.priorities()}
        ):
            lines = "\n".join(self.priority_lines())
            updates["优先级"] = (
                lines.rstrip() + ("\n" if lines.strip() else "") + candidate
            )
        if candidate and candidate != current:
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
        return candidate or current

    # HTTP 访问控制：只根据实际连接地址判断本机，忽略客户端可伪造的转发头。
    def authorized(self, request):
        peer = (
            request.transport.get_extra_info("peername") if request.transport else None
        )
        try:
            address = ipaddress.ip_address(peer[0])
            if address.is_loopback or (
                isinstance(address, ipaddress.IPv6Address)
                and address.ipv4_mapped
                and address.ipv4_mapped.is_loopback
            ):
                return True
        except (ValueError, TypeError, IndexError):
            pass
        value = request.headers.get("Authorization", "")
        scheme, _, token = value.partition(" ")
        return bool(
            self.allow_external
            and self.access_key
            and scheme.lower() == "bearer"
            and secrets.compare_digest(token.encode(), self.access_key.encode())
        )

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

    # HTTP 模型接口：查询上游并检查当前模型，失败时展示旧缓存。
    async def models(self, request):
        if not self.authorized(request):
            return self.error("Invalid access key", 401)
        try:
            await self.refresh_models()
        except Exception:
            # 查询失败沿用上次成功列表，不改写当前模型及优先级。
            pass
        try:
            return self.model_listing()
        except ConfigError as exc:
            return self.error(str(exc), 503)

    # 别名使用当前实际模型的上下文；指定模型使用自己的配置，未知长度不冒充已知值。
    def model_listing(self):
        current = self.text("当前模型")
        contexts = {}
        for name, size in self.priorities():
            contexts.setdefault(name, size)
        aliases = {MODEL_NAME, self.public_model()}
        names = list(
            dict.fromkeys(
                (MODEL_NAME, self.public_model(), *(self.available_models or []))
            )
        )
        items = []
        for name in names:
            item = {"id": name, "object": "model", "created": 0, "owned_by": "auto"}
            size = (
                contexts.get(current, DEFAULT_CONTEXT)
                if name in aliases
                else contexts.get(name)
            )
            if size is not None:
                item["context_window"] = size
            items.append(item)
        result = {
            "object": "list",
            "data": items,
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
        if not self.authorized(request):
            return self.error("Invalid access key", 401)
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

    # 路由：指定名称原样转发；已确认不存在的名称直接使用自动模型。
    async def route_model(self, requested):
        direct = (
            requested not in (MODEL_NAME, self.public_model())
            and requested not in self.missing_models
        )
        if direct:
            return requested, True
        current = self.text("当前模型")
        return current or await self.select(""), False

    # 自动路由失败时仅在选出不同模型后重试；指定路由只处理明确的模型错误。
    async def retry_model(self, current, direct, missing=False, generation=None):
        if direct:
            if not missing:
                return current
            # 分组改变时旧 auto 也可能失效；回退前先共用一次列表刷新。
            # 刷新失败仍可尝试旧 auto，会话调用总数依然最多两次。
            try:
                await self.refresh_models((current, generation))
            except Exception:
                pass
            return self.text("当前模型")
        return await self.select(current)

    # HTTP 会话入口：一次原请求最多发送两次上游会话，不递归重试。
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
        if "stream" in body and not isinstance(body["stream"], bool):
            return self.error("stream must be boolean", 400)
        try:
            current, direct = await self.route_model(body["model"])
            if not current:
                return self.error("No configured model is available", 503)
            for attempt in range(2):
                response = None
                missing, status = False, 502
                upstream_error = None
                generation = self.models_generation
                message = "Upstream request failed"
                try:
                    response = await self.request_response(dict(body, model=current))
                    status = response.status
                    if status < 300:
                        if body.get("stream", False):
                            return await self.relay_stream(request, response)
                        return await self.relay_response(response)
                    # 错误正文同样限长读取；仅第一次明确的模型错误允许回退。
                    try:
                        data = await self.limited_read(response, MAX_RESPONSE_BYTES)
                    except ValueError:
                        return self.error("Upstream response exceeds size limit", 502)
                    upstream_error = web.Response(
                        body=data,
                        status=status,
                        headers=self.response_headers(response),
                    )
                    if attempt == 0 and len(data) <= MAX_ERROR_BYTES:
                        missing = self.model_unavailable(status, data)
                    if direct:
                        # 旧请求的错误不污染刚刷新的列表对应记录。
                        if missing and generation == self.models_generation:
                            self.missing_models.add(body["model"])
                except (aiohttp.ClientError, OSError, asyncio.TimeoutError):
                    status = 502
                    message = "Upstream connection failed or response interrupted"
                finally:
                    if response is not None:
                        response.release()
                if attempt == 0 and missing:
                    try:
                        selected = await self.retry_model(
                            current, direct, missing, generation
                        )
                    except Exception:
                        selected = current
                    if selected and selected != current:
                        current, direct = selected, False
                        continue
                return (
                    upstream_error
                    if upstream_error is not None
                    else self.error(message, status)
                )
        except ConfigError as exc:
            return self.error(str(exc), 503)
        except Exception:
            return self.error("Auto service configuration or upstream error")

    # 流式响应边收边转发；一旦开始返回响应，就不再进入重试流程。
    async def relay_stream(self, request, response):
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
        except (aiohttp.ClientError, ConnectionError, OSError, asyncio.TimeoutError):
            downstream.force_close()
            if request.transport:
                request.transport.close()
        return downstream

    # 普通响应限制读取大小，只透传必要响应头。
    async def relay_response(self, response):
        try:
            data = await self.limited_read(response, MAX_RESPONSE_BYTES)
        except ValueError:
            return self.error("Upstream response exceeds size limit", 502)
        return web.Response(
            body=data, status=response.status, headers=self.response_headers(response)
        )

    # 成功和错误共用响应头白名单，保留重试时间和请求标识。
    @staticmethod
    def response_headers(response):
        return {
            name: response.headers[name]
            for name in ("Content-Type", "X-Request-Id", "Retry-After")
            if name in response.headers
        }

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
                else ("otto" if random.random() < 0.01 else MODEL_NAME)
            )
            model_id = (
                existing_model["id"] if existing_model else f"{source_id}/{chosen_name}"
            )
            if existing_model is None and any(
                p.get("id") == model_id for p in config.get("provider", [])
            ):
                raise RuntimeError("Model ID already belongs to a different model")
            await self.ensure_source(config, source_id)
            await self.ensure_provider(manager, existing_model, model_id, chosen_name)

    # 创建或补齐供应商，只更新插件拥有的本机地址和缺失字段。
    async def ensure_source(self, config, source_id):
        from astrbot.dashboard.services.config_service import ProviderConfigService

        service = ProviderConfigService(
            SimpleNamespace(
                astrbot_config=config, provider_manager=self.context.provider_manager
            )
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
                    "api_base": self.local_base,
                    "key": ["auto"],
                    "enable": True,
                    "timeout": 120,
                    "proxy": "",
                    "custom_headers": {},
                },
            )
        else:
            previous_url = urlsplit(source.get("api_base") or "")
            # 只迁移插件自身的本机供应商地址，保留其他自定义设置。
            owned_local_source = (
                previous_url.scheme == "http"
                and previous_url.hostname == "127.0.0.1"
                and previous_url.path.rstrip("/") == "/v1"
                and not previous_url.username
                and not previous_url.password
                and not previous_url.query
                and not previous_url.fragment
            )
            if source.get("type") != "openai_responses" or not owned_local_source:
                raise RuntimeError(
                    "Provider source auto already exists with different settings"
                )
            updated_source = dict(source)
            updated_source["api_base"] = self.local_base
            # 本机免密，但客户端仍需要占位 key；只补齐缺失或全空的配置。
            keys = source.get("key")
            if not keys or (
                isinstance(keys, list)
                and not any(isinstance(key, str) and key.strip() for key in keys)
            ):
                updated_source["key"] = [MODEL_NAME]
            updated_source.setdefault("proxy", "")
            updated_source.setdefault("custom_headers", {})
            if updated_source != source:
                await service.upsert_provider_source(source_id, updated_source)

    # 创建或补齐模型，并保留用户已编辑的能力、上下文和请求体参数。
    async def ensure_provider(self, manager, existing_model, model_id, chosen_name):
        source_id = MODEL_NAME
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
                "max_context_tokens": DEFAULT_CONTEXT,
            }
        )

    # 按开关绑定本机或全部网卡，注册模型仍指向本机地址。
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
            await web.TCPSite(
                self.runner,
                "0.0.0.0" if self.allow_external else "127.0.0.1",
                self.port,
            ).start()
            if getattr(
                getattr(self.context, "provider_manager", None), "provider_insts", None
            ):
                await self.ensure_model()
        except Exception:
            await self.terminate()
            raise

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
