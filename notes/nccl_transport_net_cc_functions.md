# NCCL `src/transport/net.cc` 函数手册

> **文件**：`src/transport/net.cc`（约 2133 行）  
> **目的**：按函数说明 **具体做什么、在什么线程、何时被调用**。  
> **入门**：shared / 非 shared、设计动机见 [`nccl_transport_net_cc_beginner_guide.md`](./nccl_transport_net_cc_beginner_guide.md)  
> **结构总览**：[`nccl_transport_net_cc_analysis.md`](./nccl_transport_net_cc_analysis.md)  
> **术语**：本文 **P2P = 走 NET 的 send/recv 点对点**；**graph = `ncclTopoGraph*`（拓扑算法图）**，不是 CUDA Graph、不是本机 GPU peer。

---

## 目录

1. [怎么读本手册](#1-怎么读本手册)
2. [函数总表](#2-函数总表)
3. [宏与数据结构（不是函数，但是函数都在用）](#3-宏与数据结构不是函数但是函数都在用)
4. [连接判定与 netAttr](#4-连接判定与-netattr)
5. [主线程：Setup / Connect / Free](#5-主线程setup--connect--free)
6. [辅助：SHM / dump / DMA-BUF 标志](#6-辅助shm--dump--dma-buf-标志)
7. [共享缓冲池](#7-共享缓冲池)
8. [Proxy：Setup / Connect / Free](#8-proxysetup--connect--free)
9. [Proxy：Progress（运行时核心）](#9-proxyprogress运行时核心)
10. [用户缓冲注册 / 注销](#10-用户缓冲注册--注销)
11. [传输表 `netTransport`](#11-传输表-nettransport)
12. [调用关系简图](#12-调用关系简图)
13. [修订记录](#13-修订记录)

---

## 1. 怎么读本手册

每个函数尽量给出：

| 栏 | 含义 |
|----|------|
| **线程** | 主线程（init/enqueue 侧）还是 Proxy 线程 |
| **角色** | 建连 / 进度 / 注册 / 辅助 |
| **做什么** | 用白话说完这一函数的职责 |
| **要点** | 容易漏的细节 |

vtable 里挂上的函数（`setup/connect/progress/...`）是 **对外接口**；其余多为内部实现。

---

## 2. 函数总表

| 行号（约） | 函数 | 线程 | 一句话 |
|------------|------|------|--------|
| 160 | `canConnect` | 主 | 这对 peer 能不能走 NET |
| 192 | `populateCommNetAttrs` | 主 | 按 P2P/集体填插件 hint |
| 221 | `setNetAttrs` | Proxy | 把 hint 交给插件 `setNetAttr` |
| 228 | `printNetAttrs` | 任意 | TRACE 打印 hint |
| 249 | `setXferNetAttrs` | Proxy | 按本次 op 收紧 hint |
| 285 | `getHandleForAddressRangeFlags` | Proxy | DMA-BUF 是否强制 PCIe |
| 296 | `sendSetup` | 主 | 发送侧选网卡/PXN，RPC Setup |
| 342 | `ncclTopoFlushTypeStr` | 调试 | flush 类型转字符串 |
| 356 | `recvSetup` | 主 | 接收侧 listen，拿 handle |
| 402 | `netMapShm` | 主 | 跨进程导入 HOST 映射 |
| 408 | `netCreateShm` | Proxy | 跨进程创建 HOST SHM |
| 414 | `netDumpMap` | 调试 | 打印 connectMap |
| 450 | `sendConnect` | 主 | 异步 Connect + 填 GPU conn |
| 563 | `recvConnect` | 主 | 同上（收侧） |
| 636 | `sendFree` | 主 | 释放主线程侧 send map |
| 660 | `recvFree` | 主 | 释放主线程侧 recv map |
| 666 | `sharedNetBuffersInit` | Proxy | 建/引用 P2P 共享池 |
| 706 | `sharedBuffersGet` | Proxy | 池内 offset |
| 715 | `sharedNetBuffersDestroy` | Proxy | 减引用，必要时释放池 |
| 745 | `proxySharedInit` | Proxy | NVB 预连接时先建池 |
| 752 | `sendProxySetup` | Proxy | 分配 sendNetResources |
| 793 | `recvProxySetup` | Proxy | 分配 recv 资源 + **listen** |
| 836 | `ncclNetGetDeviceHandle` | Proxy | 是否要 GPU 侧 net handle |
| 858 | `sendProxyConnect` | Proxy | **connect** + 分配缓冲 + regMr |
| 1043 | `recvProxyConnect` | Proxy | **accept** + 分配缓冲 + regMr |
| 1216 | `sendProxyFree` | Proxy | 关 send、dereg、释放内存 |
| 1271 | `recvProxyFree` | Proxy | 关 recv、dereg、释放内存 |
| 1324 | `sendProxyProgress` | Proxy | 发送流水线：credit → isend → test |
| 1493 | `recvProxyProgress` | Proxy | 接收流水线：irecv → test → flush |
| 1790 | `ncclNetDeregBuffer` | 主 | RPC 注销用户缓冲 |
| 1796 | `netRegisterBuffer` | 主 | 对每个 peer 注册用户缓冲 |
| 1863 | `ncclNetLocalRegisterBuffer` | 主 | 普通 commRegister 入口 |
| 1900 | `cleanupNet` | 主 | Graph 结束时 deregister |
| 1907 | `ncclNetGraphRegisterBuffer` | 主 | CUDA Graph 捕获期注册 |
| 1944 | `sendProxyRegBuffer` | Proxy | 对 sendComm `regMr` |
| 2011 | `recvProxyRegBuffer` | Proxy | 对 recvComm `regMr` |
| 2078 | `netHandleCmp` | Proxy | 队列里按 handle 比较 |
| 2082 | `sendProxyDeregBuffer` | Proxy | send 侧 `deregMr` |
| 2104 | `recvProxyDeregBuffer` | Proxy | recv 侧 `deregMr` |
| 2126 | `netTransport` | — | 填进全局传输表（不是函数） |

前向声明：`sendProxyProgress`（约 282 行）、`recvProxyProgress`（约 560 行），实现见后。

---

## 3. 宏与数据结构（不是函数，但是函数都在用）

文件前部定义 **connectMap 银行** 与 **send/recvNetResources**，几乎每个函数都碰它们。

- **银行**：HOST / DEV / SHARED_HOST / SHARED_DEV / GDC  
- **offset 编码**：哪块银行 + 偏移；`NCCL_NET_MAP_GET_POINTER` 解出 CPU/GPU 指针  
- **`setupReq`**：Setup RPC 载荷  
- **`netRegInfo`**：注册 RPC 载荷（地址、大小、物理段数）  

详见 beginner / analysis 文档。

---

## 4. 连接判定与 netAttr

### `canConnect`（约 160）

- **线程**：主  
- **做什么**：判断 `info1` 与 `info2` 是否允许用 NET。  
- **逻辑**：默认可以；**同一 host** 时问拓扑 `ncclTopoCheckNet`（节点内可能禁用 NET，改走 SHM/P2P transport）。  
- **不做什么**：不选网卡、不建连。

### `populateCommNetAttrs`（约 192）

- **线程**：主（Connect 前）  
- **做什么**：填 `ncclNetAttr_t`，告诉插件大概会有多少并发 peer / 每 peer 多少流。  
- **P2P 连接**（`conn->p2pOnly`）：按 `p2pnChannels`、send/recv/AlltoAll 等位图。  
- **集体连接**：按 tree 度数、`nChannels`。  

### `setNetAttrs`（约 221）

- **线程**：Proxy  
- **做什么**：若插件实现了 `setNetAttr`，把当前 hint 设进去，并缓存在 `proxyState->netAttr`。

### `printNetAttrs`（约 228）

- **做什么**：把 op/algo/proto 位图转成字符串，`TRACE(NCCL_NET, ...)`。纯调试。

### `setXferNetAttrs`（约 249）

- **线程**：Proxy progress  
- **做什么**：按 **这一次** `args` 的 nPeers/nChannels/算法/协议收紧 hint；有变化才 `setNetAttrs`。  
- **含义**：插件可按真实负载调内部资源，而不是按建连时的上限一直开着。

---

## 5. 主线程：Setup / Connect / Free

### `sendSetup`（约 296）

- **线程**：主（`selectTransport` → `transportComm->setup`）  
- **做什么（发送侧建连第一步）**：  
  1. 决定 `shared`：有 **topo graph** 或 `connIndex==0` → 0；否则默认网络 P2P 用 1。  
  2. `ncclTopoGetNetDev`：选网卡、**proxyRank**（可能 PXN，即别的本地 rank 出网）。  
  3. `ncclTopoCheckGdr`：能否 GDR；置 `NCCL_DIRECT_NIC`。  
  4. `ncclProxyConnect(TRANSPORT_NET, send, proxyRank)`。  
  5. **阻塞** `ProxyMsgSetup`，在 proxy 上创建 `sendNetResources`。  
  6. `connectInfo` 里写 **proxy 的 top-parent rank** + **useGdr**，给对端 recv。  

### `recvSetup`（约 356）

- **做什么（接收侧建连第一步）**：  
  - 选网卡时 peer 用 **自己**（本 rank 的 NIC）。  
  - **不支持 PXN on recv**：proxy 必须是本 rank。  
  - GDR 时算 `needFlush`。  
  - Setup 的 **响应是 `ncclNetHandle_t`**（listen 句柄），再附 useGdr。  
  - 对端 send 用这个 handle 去 `connect`。  

### `ncclTopoFlushTypeStr`（约 342）

- **做什么**：把 `needFlush` 枚举打成可读字符串，给日志用。

### `sendConnect`（约 450）

- **线程**：主  
- **做什么**：完成发送侧连接，让 GPU 能用 `send->conn`。  
  1. 读对端 useGdr；对端没有 GDR 则清掉本端 `DIRECT_NIC`。  
  2. 第一次：分配 `connectMap`，`ProxyCallAsync(Connect)`，把对端 handle + netAttr 发给 proxy。  
  3. `PollProxyResponse` 拿到 proxy 填好的 map（可 `ncclInProgress` 再进）。  
  4. 跨进程/跨 GPU：import SHM、CUDA IPC；必要时 peerAccess。  
  5. 从 map 解出 GPU 指针，填 `head/tail/connFifo/buffs/stepSize`。  
  6. 挂 `proxyProgress = sendProxyProgress`（设备网路径且不需 proxy 时可为 NULL）。  

### `recvConnect`（约 563）

- **做什么**：与 send 对称。  
  - RPC 载荷主要是 **对端 send 的 proxyRank** + netAttr。  
  - 填 `recv->conn`（head/tail 与 send 交叉映射）。  
  - 挂 `recvProxyProgress`。  

### `sendFree`（约 636）

- **做什么**：主线程释放 **自己这份** send 的 `connectMap`。  
  - 若缓冲在别的 GPU：关 IPC / cuMem。  
  - 跨进程：关 SHM attach。  
  - **不**关插件连接（那是 `sendProxyFree`）。  

### `recvFree`（约 660）

- **做什么**：主线程 `free(recv->transportResources)`（map 副本）。Recv 侧 proxy 与 GPU 同进程，无需复杂 IPC teardown。

---

## 6. 辅助：SHM / dump / DMA-BUF 标志

### `netMapShm`（约 402）

- **主线程**：用 proxy 创建的 shareable desc，**导入** HOST 缓冲到本进程 CPU/GPU 指针（PXN：kernel 与 proxy 不同进程）。

### `netCreateShm`（约 408）

- **Proxy**：按 map 里 HOST 大小 **创建** 可导出的 SHM，填 `cpuPtr/gpuPtr/createDesc`。

### `netDumpMap`（约 414）

- **调试**：printf 各银行大小、sendMem/recvMem/buffs 指针。正式路径注释掉。

### `getHandleForAddressRangeFlags`（约 285）

- **做什么**：`cuMemGetHandleForAddressRange` 的 flags。  
- CUDA≥12.8 且 GDR 标成 **PCI** 时，强制 DMA-BUF 走 **PCIe**，避免在 C2C+PCI 机器上绑错总线。

---

## 7. 共享缓冲池

（仅 **网络 P2P 的 shared buffers**，见 beginner 文档。）

### `sharedNetBuffersInit`（约 666）

- **线程**：Proxy  
- **做什么**：为某 `tpLocalRank` 的 send 或 recv 建/引用一块池：  
  `size = nChannels × NCCL_SHARED_STEPS(16) × p2pChunkSize`  
- `refcount++`；第一次才真正分配 CUDA 或 host。  
- 跨进程 send 不允许 “host 数据缓冲” 的某种组合（会 WARN）。  

### `sharedBuffersGet`（约 706）

- **做什么**：`(channel, slot) → offset = p2pChunkSize * (channel*16 + slot)`。  
- Progress 里把 offset 写进 `connFifo`，GPU 写到「池基址 + offset」。

### `sharedNetBuffersDestroy`（约 715）

- **做什么**：`refcount--`，到 0 则释放 CUDA/host 与 IPC；再清 `localPeers` 空项。

### `proxySharedInit`（约 745）

- **做什么**：vtable 的 `proxySharedInit`。NVB 预连接时先把 **device 共享池** 建起来（`type=0` send，cuda=1）。  
- 此时连接可能还没 `connect`，`sendProxyFree` 里用 `connSharedInitialized` 只拆池、不关 netComm。

---

## 8. Proxy：Setup / Connect / Free

### `sendProxySetup`（约 752）

- **RPC**：`ProxyMsgSetup`  
- **做什么**：`calloc sendNetResources`，抄 `setupReq`，`getProperties`：  
  - DMA-BUF 是否可用、`maxRecvs`、device net 类型/版本、`maxP2pBytes`（非法则报错）  
- **不** listen/connect。响应体为空。

### `recvProxySetup`（约 793）

- **做什么**：同样填 `recvNetResources`。  
- **额外**：`ncclNet->listen(...)`，handle 写进响应 → 主线程 → bootstrap → 对端 send。

### `ncclNetGetDeviceHandle`（约 836）

- **做什么**：判断要不要给插件一张 **GPU 侧** `ncclNetDeviceHandle`。  
- 当前规则：类型为 `UNPACK`、版本匹配、且是 **recv** 才分配。  
- 否则 `*handle = NULL`（纯 host proxy 路径）。

### `sendProxyConnect`（约 858）

- **RPC**：`ProxyMsgConnect`（可多次，直到 `done=1`）  
- **做什么（发送侧真正建网 + 准备货位）**：  
  1. `setNetAttrs`。  
  2. `ncclNet->connect(handle)`；shared+SHARED_COMMS 时按 netDev×远端 rank **复用** sendComm。  
  3. `netSendComm==NULL` → `InProgress`（异步 connect）。  
  4. 按 shared/非 shared **分配协议缓冲** + sendMem/recvMem。  
  5. 跨进程 SHM / 同进程 host alloc / GDRCopy `gdcSync`。  
  6. 初始化 `head`（shared 时先 `-NCCL_STEPS`）、`connFifo[].size=-1`。  
  7. 对各协议缓冲 `regMr` 或 `regMrDmaBuf`。  
  8. 把整份 `connectMap` 拷回主线程。  

### `recvProxyConnect`（约 1043）

- **做什么**：  
  1. 记下对端 `proxyRank`。  
  2. `ncclNet->accept`（shared comm 则复用 recvComm）。  
  3. accept 未完成 → InProgress。  
  4. **立刻 `closeListen`**（这条 listen 已用完）。  
  5. **禁止 remote proxy**：`sameProcess==0` 直接内部错误（recv 不做 PXN）。  
  6. 分配缓冲、host 控制面、可选 `gdcSync/gdcFlush`。  
  7. `regMr` / DMA-BUF，返回 map。  

### `sendProxyFree`（约 1216）

- **做什么**：  
  - `connSharedInitialized`：只拆共享池（预连接）。  
  - `connConnected`：用户 MR 队列全部 `deregMr`；协议缓冲 dereg；释放 HOST/DEV/GDC；shared 池减引用；`closeSend`（shared comm 看 refcount）。  
  - `free(resources)`。  

### `recvProxyFree`（约 1271）

- **做什么**：与 send 对称；`closeRecv`；shared comm 用 `tpRemoteProxyRank` 索引。

---

## 9. Proxy：Progress（运行时核心）

每次 collective/p2p 由 `proxy.cc` 调 `args->progress`，即下面两个函数。

### `sendProxyProgress`（约 1324）

- **线程**：Proxy  
- **状态**：`Ready` → 初始化各 sub 的 `base/step`、清 posted/transmitted/done → `Progress`。  

**Progress 对每个未完成的 sub：**

| 阶段 | 条件（简化） | 具体事情 |
|------|----------------|----------|
| **posted** | 还有步且未超出流水线深度 | shared：领池子 offset 写入 fifo，**提前**加 head（给 GPU 空位）；非 shared：只加计数 |
| **transmitted** | GPU 已写好（size≠-1 且 tail 够；LL 扫 flag） | `setXferNetAttrs`；`ncclNet->isend`；request 非空则 transmitted++ |
| **done** | 已 isend 未完成 | `test`；成功则 size=-1、fence、非 shared 更新 head；全完成则 `args->done++` |

全部 sub 完成 → `args->state = None`。  
`idle=1` 表示这一轮没推进，proxy 可去做别的 op。

### `recvProxyProgress`（约 1493）

- **Ready**：把相同 `netRecvComm` 的 sub **排成一组**（`groupSize ≤ maxRecvs`），一次 irecv 多路。  

**Progress：**

| 阶段 | 具体事情 |
|------|----------|
| **posted** | 组好 ptrs/sizes/tags；SIMPLE+reg 要等 kernel 启动（fifo size==-1）；`irecv`；可选 `OPTIONAL_RECV_COMPLETION` |
| **received** | `test` 整组完成；清 size；若 GDR 需 flush 则进入 flush |
| **flush** | `gdcFlush`：x86 `mfence`+哑读；或 `iflush` |
| **后续** | 通知 GPU 可消费，推进计数直到 `done==nsteps` |

---

## 10. 用户缓冲注册 / 注销

让 NIC 直接 DMA 用户 GPU 缓冲（需 `NCCL_DIRECT_NIC`），少走中转区。

### `netRegisterBuffer`（约 1796）

- **主线程**  
- **做什么**：对每个 peer connector：  
  - 已有 handle 则复用；  
  - 否则 `ProxyMsgRegister(netRegInfo)`；成功则挂到 `regRecord->netHandleHead`。  
- 无 GDR 或注册失败 → `outRegBufFlag=0`（回退 staging）。

### `ncclNetLocalRegisterBuffer`（约 1863）

- **入口**：普通 `ncclCommRegister` 路径。  
- 查 reg 记录、合法性、物理段数（多段且不允许 multi-segment 则放弃），再调 `netRegisterBuffer`。

### `ncclNetGraphRegisterBuffer`（约 1907）

- **入口**：CUDA Graph 捕获期。  
- `ncclCommGraphRegister` + `netRegisterBuffer`；成功则往 cleanup 队列塞 `cleanupNet`。

### `cleanupNet`（约 1900）

- Graph 销毁/回放结束回调：`ncclCommGraphDeregister` + free 回调对象。

### `ncclNetDeregBuffer`（约 1790）

- **主线程**：`ProxyMsgDeregister`，让 proxy 对插件 `deregMr`。

### `sendProxyRegBuffer` / `recvProxyRegBuffer`（约 1944 / 2011）

- **Proxy**：对 **sendComm / recvComm** 注册用户地址。  
- 优先 DMA-BUF；失败或未开则 `regMr`。  
- **多物理段且无 DMA-BUF**：不注册（非 DMA-BUF 不支持多段）。  
- handle 入 `proxyMemHandleQueue`，返回给主线程。

### `netHandleCmp`（约 2078）

- 队列删除时比较两个 `proxyMemHandle` 是否同一 `handle`。

### `sendProxyDeregBuffer` / `recvProxyDeregBuffer`（约 2082 / 2104）

- 从队列摘掉对应 handle，`deregMr`。

---

## 11. 传输表 `netTransport`

文件末尾（约 2126）把上述函数填进 `ncclTransport`：

```text
name: "NET"
canConnect
send: setup, connect, free, proxySharedInit,
      proxySetup, proxyConnect, proxyFree, proxyProgress,
      proxyRegBuffer, proxyDeregBuffer
recv: 同上一套 recv*
```

`transport.cc` 的 `ncclTransports[]` 通过这份表调用，**不直接点函数名**。

---

## 12. 调用关系简图

```text
init / p2p setup
  canConnect
  sendSetup ──RPC──► sendProxySetup
  recvSetup ──RPC──► recvProxySetup (listen)
  bootstrap 交换 handle
  sendConnect ──async RPC──► sendProxyConnect (connect+alloc+regMr)
  recvConnect ──async RPC──► recvProxyConnect (accept+alloc+regMr)
  [可选] proxySharedInit / Local|Graph RegisterBuffer
              └──RPC──► send/recvProxyRegBuffer

运行时 proxy 循环
  sendProxyProgress  →  ncclNet->isend / test
  recvProxyProgress  →  ncclNet->irecv / test / iflush

销毁
  sendFree / recvFree          （主线程 map）
  sendProxyFree / recvProxyFree
  DeregBuffer → send/recvProxyDeregBuffer
```

---

## 13. 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：`net.cc` 每个函数的职责、线程、调用时机与要点 |
