# `sendSetup` 与 RPC Setup 全流程

> **代码**：`src/transport/net.cc` `sendSetup` / `sendProxySetup`；`src/proxy.cc` `ncclProxyConnect` / `ncclProxyCallBlocking` / `proxyProgressAsync`  
> **读者**：想搞清 “RPC Setup 到底是什么”  
> **相关**：[`nccl_transport_net_cc_beginner_guide.md`](./nccl_transport_net_cc_beginner_guide.md)、[`nccl_transport_net_cc_functions.md`](./nccl_transport_net_cc_functions.md)、[`nccl_proxy_internals.md`](./nccl_proxy_internals.md)

---

## 1. 先记住一句话

**`sendSetup` 自己并不 listen/connect 网卡。**  
它只做两件事：

1. **在本 rank 上想清楚**：这条发送边用哪块网卡、要不要 GDR、谁来当 proxy、缓冲突不共享。  
2. **让那个 proxy 进程/线程准备好一份 `sendNetResources` 空壳**（记下上述决定，查插件属性）。  

第 2 步就是文档里说的 **RPC Setup**：主线程通过 **socket 控制通道** 给 **proxy 服务线程** 发一条类型为 `ncclProxyMsgSetup` 的请求，proxy 调用 `sendProxySetup`，做完再回一个“完成”。

真正的 `ncclNet->connect` 发生在后面的 **RPC Connect**（`sendConnect` → `sendProxyConnect`），不在 Setup。

```text
sendSetup          = 决策 + 通知 proxy「开一个发送连接对象」
sendProxySetup     = proxy 上执行：calloc 资源、getProperties
sendConnect        = 下一阶段：用对端 listen handle 真正 connect + 分配缓冲
```

---

## 2. 为什么要 RPC？谁和谁说话？

NCCL 每个 GPU 有一个 **proxy 服务**（独立线程，可能还在 **另一个 rank 的进程** 里——PXN）。

- **主线程**（跑 `ncclCommInit` / `sendSetup`）：选拓扑、交换 bootstrap 信息。  
- **Proxy 服务线程**：以后要循环 `isend`，网卡句柄、缓冲最好活在它这边。  
- 两边可能 **同进程**，也可能 **同机不同进程**（PXN：GPU0 的数据由 GPU1 的 proxy 出网）。

所以不能简单“函数直接调用 `sendProxySetup`”（跨进程就调不到）。  
NCCL 做成：**主线程是 client，proxy 是 server，中间是 Unix/TCP socket 上的请求–应答**。这就是这里的 **RPC**。

```text
主线程 (client)                         Proxy 服务线程 (server)
     │                                         │
     │  socket: 消息类型 + 请求体 + opId         │
     │────────────────────────────────────────►│
     │                                         │  tcomm->proxySetup(...)
     │  socket: 结果 + 可选响应体                │
     │◄────────────────────────────────────────│
     │  CallBlocking 才返回                     │
```

消息类型枚举（`proxy.h`）：

| 类型 | 值 | 含义 |
|------|-----|------|
| `ncclProxyMsgInit` | 1 | 先在 proxy 上登记一条 connection（谁、哪个 transport、send 还是 recv） |
| `ncclProxyMsgSetup` | **3** | **本文：传输自己的 setup**（NET 即 `sendProxySetup`） |
| `ncclProxyMsgConnect` | 4 | 下一步：插件 connect/accept + 分配缓冲 |

`sendSetup` 里会先 **Init**（在 `ncclProxyConnect` 内部），再 **Setup**。

---

## 3. 谁在什么时候调用 `sendSetup` / `recvSetup`？

### 3.1 是不是「主线程 bootstrap 阶段」？

**不完全是。** 拆开说：

| 说法 | 对不对 |
|------|--------|
| **主线程**调用（init / 第一次建边的那个线程） | **对**；不是 proxy 进度线程 |
| 属于最早的 **`bootstrapInit` 握手**（uniqueId / 根进程） | **不对** |
| 发生在 bootstrap **已经建好之后** | **对** |
| **借用 bootstrap** 把 Setup 产出的 `connectInfo` 发给对端 rank | **对** |
| **只**在 comm 初始化时调用一次 | **不对**；运行时第一次 P2P 建边还会再走 |

更准确的名字是：**传输建连（transport setup）**，不是 bootstrap 协议本身。

```text
ncclCommInitRank
  ├─ ① bootstrapInit              ← 真正的 “bootstrap 阶段”
  │     带外 TCP、交换 uniqueId / 地址
  │
  ├─ ② initTransportsRank
  │     AllGather 拓扑、算 ring/tree
  │     起 proxy 线程
  │
  └─ ③ 建 Ring/Tree/NVLS/P2P 边
        ncclTransportP2pSetup(comm, graph, connIndex)
          selectTransport
            → sendSetup / recvSetup     ← 这里（主线程）
          bootstrapSend / Recv          ← 用 ① 的通道交换 connectInfo
          sendConnect / recvConnect
```

`ncclTransportP2pSetup`（`src/transport.cc`）里顺序就是：先 `selectTransport` → `setup`，再 `bootstrapSend`/`bootstrapRecv` 交换 `ncclConnect`。

### 3.2 两种「带外通道」不要混

`sendSetup`/`recvSetup` 里会碰到 **两条完全不同的带外路径**：

| 通道 | 谁跟谁 | 干什么 |
|------|--------|--------|
| **Proxy RPC**（Init / Setup / Connect） | 本 rank **主线程** ↔ **本机（或 PXN）proxy** | 在 proxy 上建 `sendNetResources` / listen |
| **Bootstrap** | 本 rank ↔ **对端 rank** | 交换 handle、proxyRank、useGdr |

RPC Setup **不是** bootstrap；它发生在 bootstrap 已经可用之后，并且 Setup 返回后才用 bootstrap 把结果送给对端。

### 3.3 调用栈（集体拓扑边，init 时）

```text
initTransportsRank
  → ncclTransportRingConnect / TreeConnect / …
  → ncclTransportP2pSetup(comm, &graphs[RING|TREE|…], connIndex=0)
  → selectTransport(..., graph, channel, peer, connIndex)
  → 选中 NET 后
  → netTransport.send.setup  == sendSetup(...)
  → netTransport.recv.setup  == recvSetup(...)
```

### 3.4 不只在 comm init：运行时还会再调

| 时机 | 典型调用 | graph |
|------|----------|--------|
| init 里集体边 | `ncclTransportP2pSetup(comm, &graphs[RING/TREE], 0)` | **非 NULL**（topo 算法图） |
| init 里 NVB 预连接 | `ncclTransportP2pSetup(comm, NULL, 1)`（`init.cc`） | NULL |
| 运行时第一次 send/recv（runtime connect） | `group.cc` / enqueue `ncclTransportP2pSetup(comm, NULL, 1)` | NULL |
| CollNet / NVLS tree 建边 | 同样 `P2pSetup` + 对应 graph | 非 NULL |

`graph==NULL` 的那些是 **网络 send/recv 点对点边**，不是集体 topo 边。

参数含义：

| 参数 | 含义 |
|------|------|
| `comm` | 本 communicator |
| `graph` | **`ncclTopoGraph*`**：集体 Ring/Tree 等拓扑图；P2P send/recv 常为 NULL |
| `myInfo` / `peerInfo` | 本端、对端 rank 信息 |
| `connectInfo` | **输出**：给对端 recv 的一小块元数据（proxy rank + useGdr） |
| `send` | 本端这条发送 connector（之后挂 `proxyConn`、`conn.flags`） |
| `channelId` | 通道号 |
| `connIndex` | 连接槽：0 多为集体主连接 |

---

## 4. `sendSetup` 逐步在干什么

源码：`net.cc` 约 296–335 行。

### 步骤 A：决定缓冲突不共享

```c
send->conn.shared = req.shared =
    graph || connIndex == 0 ? 0 :
    (用户设了 NCCL_NET_SHARED_BUFFERS ? 用户值 : 1);
```

| 情况 | shared | 白话 |
|------|--------|------|
| 带着 **topo graph** 建边（集体拓扑边） | 0 | 专用大缓冲 |
| `connIndex==0` | 0 | 主连接，同样专用 |
| 否则（网络 send/recv 点对点） | 默认 1 | 用共享池 |

`req` 是马上要发给 proxy 的结构体 `setupReq`。

### 步骤 B：选网卡和谁当 proxy

```c
ncclTopoGetNetDev(comm, myRank, graph, channelId, peerRank,
                  &netId, &req.netDev, &proxyRank);
```

- `req.netDev`：插件眼里的网卡编号。  
- `proxyRank`：由哪个 rank 的 proxy 出网。  
  - **等于自己**：本 GPU 的 proxy 发。  
  - **不等于自己（PXN）**：同机另一 GPU 的 proxy 发（通常更靠近那块 NIC）。  
- `connIndex==0` 且 PXN → `comm->useNetPXN = true`。

### 步骤 C：要不要 GDR

```c
ncclTopoCheckGdr(..., /*isSend=*/1, &req.useGdr);
if (useGdr) send->conn.flags |= NCCL_DIRECT_NIC;
if (!useGdr && connIndex==0) comm->useGdr = 0;
```

GDR = 网卡能否直接 DMA GPU 内存。集体主连接若不能 GDR，会把 comm 级 `useGdr` 关掉。

### 步骤 D：连上 proxy 服务（里面已有一次 RPC Init）

```c
ncclProxyConnect(comm, TRANSPORT_NET, /*send=*/1, proxyRank, &send->proxyConn);
```

详见下一节。结束后 `send->proxyConn` 里有：

- 到该 proxy 的 **socket**  
- proxy 上那条 **connection 指针**（对端堆里的地址，当 cookie）  
- `sameProcess`：是否同进程  

### 步骤 E：填 setupReq 其余字段

```c
req.tpLocalRank / tpRank / tpRemoteRank   // split/shrink 后的 top-parent 身份
req.sameDevice = (peerInfo[proxyRank].cudaDev == comm->cudaDev)
req.channelId, connIndex, netDev, shared, useGdr  // 前面已填
```

`sameDevice`：kernel 与 **执行发包的那个 proxy 所在 GPU** 是否同一块卡（影响后面能不能 GDRCopy 映射 head）。

### 步骤 F：RPC Setup（阻塞等到做完）

```c
ncclProxyCallBlocking(comm, &send->proxyConn,
                      ncclProxyMsgSetup,   // 消息类型 = Setup
                      &req, sizeof(req),   // 请求体 = setupReq
                      NULL, 0);            // 无响应体
```

Proxy 上执行 **`sendProxySetup`**（见 §6）。  
发送侧 Setup **不需要返回数据**（listen handle 在 **recv** 侧 Setup 才有）。

### 步骤 G：打日志 + 写 connectInfo

```c
// 日志: Channel xx : me -> peer [send] via NET/IB/dev /GDRDMA /Shared (proxyRank)
*((int*)connectInfo) = topParentRanks[proxyRank];
memcpy(connectInfo + sizeof(ncclNetHandle_t), &req.useGdr, sizeof(int));
```

`connectInfo` 会经 **bootstrap** 发给对端 `recvConnect`：

- 前面一部分：发送侧 **proxy 的 rank**（对端要知道是谁连过来；PXN 时不是数据 GPU 自己）。  
- 偏移 `sizeof(ncclNetHandle_t)` 处：发送侧 **useGdr**（两端协商：有一边不行就关 DIRECT_NIC）。  

注意：发送 Setup **还没有** 把 `ncclNetHandle_t` 填满；handle 来自 **对端 recvSetup 的 listen**。发送侧这块前缀先用来塞 proxyRank。

---

## 5. `ncclProxyConnect`：RPC 之前先“挂电话”

`proxy.cc` 约 1140 行。在发 Setup 之前必须有一条到 proxy 的控制连接。

```text
1. 算 sameProcess（同 host + 同 pid？）
2. 若还没有到该 tpRank 的 socket → SocketInit + Connect
   （地址来自 comm 初始化时交换的 peerAddresses）
3. RPC Init（又是一次 CallBlocking，类型 ncclProxyMsgInit）:
     请求: transport=NET, send=1, 本端 tpRank, sameProcess
     响应: proxy 上新分配的 connection* ，以及 progress 用的 shm 路径
4. 若该传输有 proxyProgress，打开共享 ops 池（后面真正发包用）
5. proxyConn->initialized = true
```

**Init 和 Setup 的差别：**

| | Init | Setup |
|--|------|--------|
| 谁处理 | `proxyConnInit`（proxy 通用） | `sendProxySetup`（NET 专用） |
| 目的 | “登记一条 NET 发送 connection” | “按 setupReq 填这份 connection 的网卡属性” |
| 结果 | `proxyConn->connection` cookie | `connection->transportResources = sendNetResources` |

---

## 6. `ncclProxyCallBlocking` + proxy 如何跑到 `sendProxySetup`

### 6.1 主线程：Blocking = Async + 空转 Poll

```c
// proxy.cc ncclProxyCallBlocking
malloc 一个 opId 当本次请求的票据
ncclProxyCallAsync(..., type=Setup, req=&setupReq, respSize=0, opId)
do {
  res = ncclPollProxyResponse(..., opId);
} while (res == ncclInProgress);
```

`CallAsync` 往 socket **写出**：connection 指针、reqSize、respSize、`setupReq` 字节、opId。

### 6.2 Proxy 服务线程：收包 → 分发

```text
proxy 循环读 socket
  认出 type == ncclProxyMsgSetup
  proxyServiceInitOp:
    收 connection*、req、opId
    入异步队列
    proxyProgressAsync:
      connection->tcomm->proxySetup(...)
        == net 的 sendProxySetup   （因为 Init 时登记的是 NET send 半边）
      done==1:
        把 connection.state = connSetupDone
        socket 回: {opId, 错误码, respSize=0}
        出队
```

主线程 `Poll` 读到对应 opId 的应答 → `CallBlocking` 返回。

### 6.3 `sendProxySetup` 具体做了什么

`net.cc` 约 752–791 行：

1. 检查 `reqSize == sizeof(setupReq)`。  
2. `calloc sendNetResources`，挂到 `connection->transportResources`。  
3. 把 req 里的 rank、netDev、shared、GDR、channel、sameDevice **抄进去**。  
4. `ncclNet->getProperties(netDev, &props)`：  
   - 能否 DMA-BUF、`maxRecvs`、device-net 类型、`maxP2pBytes`（非法则失败）。  
5. **不分配数据缓冲、不 connect 网卡。**  
6. `respSize` 必须为 0；`*done = 1`。

Setup 结束时：proxy 上有一个 **已填属性、尚未 connect 的发送资源对象**。

---

## 7. 整条时间线（从集体建边到 Setup 完成）

```text
[Rank A 主线程]                         [Rank A 的 proxy，或 PXN 时 Rank P 的 proxy]
     │                                              │
     │ sendSetup 开始                                │
     │  算 shared / netDev / proxyRank / useGdr      │
     │                                              │
     │ ncclProxyConnect ───────────────────────────►│  （若还没连过）
     │    RPC Init ────────────────────────────────►│  建 connection 对象
     │    ◄──────────────── connection* ────────────│
     │                                              │
     │ ncclProxyCallBlocking(Setup, setupReq) ─────►│
     │                                              │  sendProxySetup:
     │                                              │    分配 sendNetResources
     │                                              │    getProperties
     │    ◄────────────── 完成（无 payload）─────────│
     │                                              │
     │ 写 connectInfo (proxyRank + useGdr)           │
     │ sendSetup 返回                                │
     │                                              │
     ▼                                              ▼
  sendSetup 返回后（仍在 ncclTransportP2pSetup 里）
  主线程 bootstrapSend/Recv：和对端交换 connectInfo
  再 sendConnect → RPC Connect → 真正 ncclNet->connect
```

对端 **recvSetup** 是对称的另一条 RPC Setup，但 **recvProxySetup 会 listen**，响应里带 `ncclNetHandle_t`。发送侧 Setup **没有** handle。

注意：图里的 bootstrap 交换发生在 **Setup 返回之后**，且 bootstrap 通道在更早的 `bootstrapInit` 里已经建好。

---

## 8. 和后面 Connect 的衔接（避免只看到半截）

| 阶段 | 主线程 | Proxy | 网卡插件 |
|------|--------|-------|----------|
| Init | `ncclProxyConnect` | 登记 connection | 无 |
| **Setup（本文）** | `sendSetup` → CallBlocking(Setup) | `sendProxySetup` | **仅 getProperties** |
| （对端）Setup | `recvSetup` | `recvProxySetup` | **listen** → handle |
| **Bootstrap 交换**（对端 rank） | `bootstrapSend/Recv` | — | — |
| Connect | `sendConnect` → CallAsync(Connect) | `sendProxyConnect` | **connect** + 分配缓冲 + regMr |

所以：**RPC Setup ≠ 连上网卡**；是 **“在正确的 proxy 上创建并配置发送资源”**。

---

## 9. 常见疑问

**Q：同进程为什么还要 RPC，不能直接调 `sendProxySetup`？**  
为了和 PXN/跨进程走同一套代码；同进程也是本机 socket，开销可接受。

**Q：Blocking 会卡住吗？**  
会等到 proxy 做完 Setup。Setup 很快（calloc + getProperties）。后面 Connect 才可能 `InProgress` 多次 poll。

**Q：PXN 时 RPC 发给谁？**  
`proxyRank` 那个 rank 的 proxy。`setupReq.sameDevice` 比较的是 **那块 GPU** 和本 comm 的 cudaDev。

**Q：失败了怎么办？**  
`CallBlocking` 返回错误；connection 可能停在未 Setup 完。上层 init 失败，不会进入正常 isend。

**Q：sendSetup/recvSetup 是主线程 bootstrap 阶段调的吗？**  
主线程 **对**；**bootstrapInit 那个阶段不对**。它们在 bootstrap 已经起来之后的 **`ncclTransportP2pSetup`（传输建连）** 里调用，并用 bootstrap **交换** Setup 的产物。运行时第一次 P2P 建边还会再调。详见 [§3](#3-谁在什么时候调用-sendsetup--recvsetup)。

---

## 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：sendSetup 逐步说明；RPC Setup 的 client/server、Init vs Setup vs Connect |
| 2026-07-10 | §3：澄清主线程 vs bootstrapInit；Proxy RPC vs bootstrap 交换；init 与 runtime 多次调用 |
