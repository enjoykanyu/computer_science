# 聊天助手bot
## openclaw
### 网关gateway层级
<font style="color:rgb(51, 51, 51);">网关是 OpenClaw 的中心控制平面，负责管理 WebSocket 连接、分发 RPC 方法、协调消息通道、编排代理运行以及维护系统状态。它作为一个单一的复用进程同时服务协议客户端（CLI、原生应用、控制界面）和通道提供者。</font>

<font style="color:rgb(51, 51, 51);">其中gateway采用了</font>**<font style="color:rgb(51, 51, 51);">统一的服务器架构</font>**<font style="color:rgb(51, 51, 51);">来同时处理WebSocket控制平面和HTTP API请求。传统的多进程架构会导致复杂的进程间通信和状态同步问题，而Gateway采用</font>**<font style="color:rgb(51, 51, 51);">单一进程多路复用</font>**<font style="color:rgb(51, 51, 51);">设计，在同一个端口上同时支持WebSocket实时通信和HTTP REST API，大大简化了系统复杂度并提高了性能</font>

##### <font style="color:rgb(51, 51, 51);">实现细节</font>
+ <font style="color:rgb(51, 51, 51);">1.1，首先通过</font>`**<font style="color:rgb(51, 51, 51);">loadServerImpl()</font>**`<font style="color:rgb(51, 51, 51);">延迟加载服务器实现</font>

```typescript
export async function startGatewayServer(
  ...args: Parameters<typeof import("./server.impl.js").startGatewayServer>
  ): ReturnType<typeof import("./server.impl.js").startGatewayServer> {
  const mod = await loadServerImpl();
return await mod.startGatewayServer(...args);
}
```

其次通过  return await mod.startGatewayServer(...args)

+ 1.2，实际启动服务器

```typescript
export async function startGatewayServer(
  port = 18789,
  opts: GatewayServerOptions = {},
): Promise<GatewayServer> {
  bootstrapGatewayNetworkRuntime();

```

+ 1.3，启动过程

启动过程：cliRun->startloop->runGatewayLoopWithSupervisedLockRecovery(startLoop)->startGatewayServer

startLoop函数

```typescript
const startLoop = async () =>
    await runGatewayLoop({
      runtime: defaultRuntime,
      lockPort: port,
      healthHost,
      start: async ({ startupStartedAt } = {}) =>
        await startGatewayServer(port, {
          bind,
          auth: authOverride,
          tailscale: tailscaleOverride,
          startupStartedAt,
        }),
    });
```

这里启动的

+ 2.1，服务器启动初始化

1，ensureOpenClawCliOnPath(); 确保cli工具可以使用

```typescript
export function ensureOpenClawCliOnPath(opts: EnsureOpenClawPathOpts = {}) {
  if (isTruthyEnvValue(process.env.OPENCLAW_PATH_BOOTSTRAPPED)) {
    return;
  }
  process.env.OPENCLAW_PATH_BOOTSTRAPPED = "1";

  const existing = opts.pathEnv ?? process.env.PATH ?? "";
  const { prepend, append } = candidateBinDirs(opts);
  if (prepend.length === 0 && append.length === 0) {
    return;
  }

  const merged = mergePath({ existing, prepend, append });
  if (merged) {
    process.env.PATH = merged;
  }
}
```

2，resolveMediaCleanupTtlMs 初始化配置

3，GatewayServerOptions 绑定服务地址

4，attachGatewayWsHandlers 引入websocket



3.1，安全认证链接



4.1，注册gateway节点


# gateway网关层级框架

作用：作为门卫校验权限，进行功能转发，再回传结论

