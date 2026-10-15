# auto模型切换
为 AstrBot 提供固定的auto模型，在同一个key切换分组后自动匹配新模型。
也就节省几十秒。

## 使用
加载后自动注册auto模型，AI那边需要手动配置一次。
在配置界面填个API和key，优先级缺什么手动填。

上游需要支持 OpenAI 兼容的 `/v1/models` 和 `/v1/responses` 接口。
插件默认监听 `127.0.0.1:56751`，需要该端口空闲，且运行 AstrBot 的账户具有绑定权限。

Local HTTP port is configurable with `listen_port` (default: `56751`, range: 1-65535). Reload the plugin or restart AstrBot after changing it. The auto provider URL is updated automatically. Ports above 1023 do not require low-port binding privileges.
