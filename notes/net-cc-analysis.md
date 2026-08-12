# NCCL `src/transport/net.cc` 源码详解

> **文件**：`src/transport/net.cc`（约 2092 行）  
> **角色**：NET 传输层宿主实现——把 **GPU kernel ↔ Proxy ↔ `ncclNet` 插件（IB/Socket 等）** 粘合起来  
> **版本语境**：NCCL 2.30.x  
> **小白入门（shared/设计动机）**：[`net-cc-beginner-guide.md`](net-cc-beginner-guide.md)  
> **相关笔记**：`proxy-internals.md`、`ib-isend-analysis.md`、`ib-send-recv-cts.md`、`net-ib-code-review.md`、`chunking-pattern-fifo-registration.md`、`thread-model.md`

---

## 目录

1. [文件定位与在架构中的位置](#1-文件定位与在架构中的位置)
2. [传输 vtable 与对外接口](#2-传输-vtable-与对外接口)
3. [核心数据结构](#3-核心数据结构)
4. [connectMap 内存银行模型](#4-connectmap-内存银行模型)
5. [环境参数](#5-环境参数)
6. [生命周期总览](#6-生命周期总览)
7. [Setup：`sendSetup` / `recvSetup`](#7-setupsendsetup--recvsetup)
8. [Proxy Setup：`*ProxySetup`](#8-proxy-setupproxyssetup)
9. [Connect：主线程与 Proxy 协作](#9-connect主线程与-proxy-协作)
10. [GPU 可见的 `ncclConnInfo` 接线](#10-gpu-可见的-ncclconninfo-接线)
11. [共享缓冲与共享连接](#11-共享缓冲与共享连接)
12. [Progress：`sendProxyProgress` 三阶段](#12-progresssendproxyprogress-三阶段)
13. [Progress：`recvProxyProgress` 多阶段](#13-progressrecvproxyprogress-多阶段)
14. [GDR / GDRCopy / DMA-BUF / Flush](#14-gdr--gdrcopy--dma-buf--flush)
15. [用户缓冲注册（UB / Graph Register）](#15-用户缓冲注册ub--graph-register)
16. [netAttr：给插件的流控提示](#16-netattr给插件的流控提示)
17. [设备侧 Net Device Handle](#17-设备侧-net-device-handle)
18. [Free / 资源回收](#18-free--资源回收)
19. [端到端时序图](#19-端到端时序图)
20. [与 net_ib / CollNet / 其它笔记的关系](#20-与-net_ib--collnet--其它笔记的关系)
21. [阅读与调试建议](#21-阅读与调试建议)
22. [源码索引与修订记录](#22-源码索引与修订记录)

---

## 1. 文件定位与在架构中的位置

### 1.1 三层分工

```text
┌─────────────────────────────────────────────────────────────┐
│ Host API / enqueue / graph / transport 选择                   │
│   (init, enqueue, transport.cc)                               │
└───────────────────────────┬─────────────────────────────────┘
                            │ ncclTransportP2pConnect / Setup
                            ▼
┌─────────────────────────────────────────────────────────────┐
│ net.cc  ← 本文件                                              │
│   TRANSPORT_NET 的 setup/connect/progress/reg                 │
│   管理 sendMem/recvMem、协议缓冲、proxy RPC、isend/irecv      │
└───────────────┬─────────────────────────┬───────────────────┘
                │ proxy 线程              │ GPU kernel
                ▼                         ▼
┌───────────────────────────┐   ┌─────────────────────────────┐
│ ncclNet 插件              │   │ prims_simple / LL / LL128   │
│ (net_ib, socket, 第三方)  │   │ 读写 conn.head/tail/buffs   │
│ listen/connect/isend/...  │   │ 写 connFifo.size            │
└───────────────────────────┘   └─────────────────────────────┘
```

**一句话：** `net.cc` **不实现 RDMA**；它实现 **NCCL 与“网络插件”之间的标准适配层**，并驱动 **proxy 进度状态机**。

### 1.2 为何需要 proxy

GPU 不能直接调用 `libibverbs` / 通用 net 插件的用户态 API。数据路径是：

```text
Kernel 把 chunk 写入协议缓冲（或用户注册缓冲）
  → 更新 tail / connFifo
Proxy 线程看到数据就绪
  → ncclNet->isend / irecv
  → test 完成后推进 head，释放 credit 给 GPU
```

本文件实现的就是这条 **生产者（GPU）– 消费者（proxy）– 网络插件** 流水线。

### 1.3 与 `coll_net.cc` 的边界

| | `net.cc` | `coll_net.cc` |
|--|----------|---------------|
| 传输表 | `TRANSPORT_NET` | `TRANSPORT_COLLNET` |
| 插件 API | 点对点 `isend/irecv` | 集合 `iallreduce/...` |
| 典型用途 | Ring/Tree 跨节点边、P2P send/recv | SHARP 等 CollNet |

二者 **共享类似的 connectMap / head-tail / proxy 状态机思路**，但本文件是 **点对点 NET**。

---

## 2. 传输 vtable 与对外接口

文件末尾（约 2085–2092 行）注册全局传输：

```c
struct ncclTransport netTransport = {
  "NET",
  canConnect,
  {sendSetup, sendConnect, sendFree, proxySharedInit,
   sendProxySetup, sendProxyConnect, sendProxyFree,
   sendProxyProgress, sendProxyRegBuffer, sendProxyDeregBuffer},
  {recvSetup, recvConnect, recvFree, proxySharedInit,
   recvProxySetup, recvProxyConnect, recvProxyFree,
   recvProxyProgress, recvProxyRegBuffer, recvProxyDeregBuffer}
};
```

| 回调 | 线程上下文 | 职责 |
|------|------------|------|
| `canConnect` | 主线程 | 是否允许 NET（同主机看 topo 是否禁用 intra-node net） |
| `send/recvSetup` | 主线程 | 选网卡、GDR、PXN proxyRank；发 `ProxyMsgSetup` |
| `send/recvConnect` | 主线程 | 异步 `ProxyMsgConnect`；导入 map；填 `ncclConnInfo` |
| `send/recvFree` | 主线程 | 释放主线程侧 map / IPC |
| `*ProxySetup/Connect/Free/Progress` | **Proxy 线程** | 真正 listen/connect、分配缓冲、`isend/irecv` |
| `*ProxyReg/DeregBuffer` | Proxy 线程 | 用户缓冲 `regMr` / `deregMr` |
| `proxySharedInit` | Proxy | 预初始化 P2P 共享缓冲 |

**另导出给其它模块的符号：**

- `setNetAttrs` / `setXferNetAttrs` / `printNetAttrs` / `populateCommNetAttrs`
- `ncclNetLocalRegisterBuffer` / `ncclNetGraphRegisterBuffer` / `ncclNetDeregBuffer`

---

## 3. 核心数据结构

### 3.1 `sendNetResources` / `recvNetResources`（Proxy 侧）

Proxy 在 `*ProxySetup` 中分配，挂在 `connection->transportResources`。

**共同字段语义：**

| 字段 | 含义 |
|------|------|
| `map` | 缓冲与控制结构的银行布局（见 §4） |
| `netSendComm` / `netRecvComm` | 插件连接句柄 |
| `netListenComm` | 仅 recv：listen 返回句柄 |
| `sendMem` / `recvMem` | 控制面：head/tail/connFifo |
| `buffers[PROTO]` / `mhandles[PROTO]` | 各协议数据缓冲与 MR |
| `tpRank` / `tpLocalRank` / `tpRemoteRank` | top-parent rank 空间（split/shrink 后的身份） |
| `netDev` | 逻辑网卡 index |
| `useGdr` / `useDmaBuf` / `needFlush` | GDR 与可见性 |
| `shared` | 是否 P2P 共享缓冲模式 |
| `channelId` / `connIndex` | 通道与连接索引 |
| `sameDevice` | proxy 与 kernel 是否同一 CUDA device（影响 GDRCopy） |
| `step` | 连接级步序号，跨 op 递增 |
| `maxRecvs` | 插件 multi-recv 能力 |
| `netDeviceHandle` | 可选 GPU 侧网络设备路径 |
| `maxP2pBytes` | 插件声明的最大 P2P 字节 |

**recv 独有：** `tpRemoteProxyRank`（对端发送侧 proxy rank，用于 PXN 路径标识）、`gdcFlush`。

### 3.2 `setupReq`（Setup RPC 载荷）

主线程 → Proxy 的 `ncclProxyMsgSetup` 参数：rank 信息、`netDev`、`useGdr`、`needFlush`、`shared`、`channelId`、`connIndex`、`sameDevice`。

### 3.3 `netRegInfo`（注册 RPC）

用户缓冲注册时传给 Proxy：`buffer`、`size`、`numSegments`（多物理段；无 DMA-BUF 时多段会失败）。

### 3.4 控制面：`ncclSendMem` / `ncclRecvMem`

虽定义在公共头文件，但在 net 路径中的用法是本文件的核心契约：

```text
GPU (kernel)                         Proxy (CPU)
─────────────                        ───────────
写 data → buff[slot]
写 connFifo[slot].size = nbytes
写 recvMem->tail++  (发布)
                                     读 size/tail
                                     isend/irecv
                                     test done
                                     size = -1
                                     sendMem->head++  (还 credit)
读 head 判断能否再用 slot
```

- **`NCCL_STEPS`（通常 8）**：环形 slot 深度  
- **`connFifo[i].size == -1`**：空槽 / 可 post recv 或 GPU 未写完  
- **shared 模式**：`connFifo[i].offset` 指示共享池中的偏移；`mode = NCCL_MODE_OFFSET`

---

## 4. connectMap 内存银行模型

### 4.1 银行类型

```c
NCCL_NET_MAP_HOSTMEM        // 0  固定主机内存（pinned）
NCCL_NET_MAP_DEVMEM         // 1  GPU 显存
NCCL_NET_MAP_SHARED_HOSTMEM // 2  共享 host 池
NCCL_NET_MAP_SHARED_DEVMEM  // 3  共享 device 池
NCCL_NET_MAP_GDCMEM         // 4  GDRCopy 映射的 tail/head
```

### 4.2 Offset 编码

每个逻辑指针（`sendMem`、`recvMem`、`buffs[p]`）编码在 32-bit offset 中：

| 位域 | 含义 |
|------|------|
| bit 31 | SHARED |
| bit 30 | DEVMEM |
| bit 29 | USED（0 表示 NULL） |
| bit 28–0 | 银行内字节偏移 |

宏：

- `NCCL_NET_MAP_ADD_POINTER`：分配并编码  
- `NCCL_NET_MAP_GET_POINTER(map, cpu|gpu, name)`：解出 CPU 或 GPU 侧指针  

**原理：** Proxy 与主线程/GPU 可能 **跨进程**（PXN：proxy 在另一 rank 进程）。  
`connectMap` 序列化后经 Proxy 响应传回，主线程再 **import SHM / CUDA IPC**，使 GPU 与 proxy 看到同一逻辑布局的不同映射。

### 4.3 典型分配策略（`sendProxyConnect`）

| 对象 | shared=0（Ring/Tree） | shared=1（P2P） |
|------|----------------------|-----------------|
| SIMPLE 缓冲 | GDR 则 DEVMEM，否则 HOST | 共享池（GDR→SHARED_DEVMEM） |
| LL 缓冲 | 常 HOST（或专用） | 可选独立 LL 缓冲 |
| LL128 | 非 LL 且 GDR 可 DEVMEM | — |
| sendMem/recvMem | 始终 HOST 侧（跨进程则 SHM） | 同左 |
| gdcSync | sameProcess + GDRCopy + sameDevice | — |

---

## 5. 环境参数

| 参数 | 默认 | 作用 |
|------|------|------|
| `NCCL_NET_SHARED_BUFFERS` | -2 | -2：P2P(`connIndex!=0` 且无 graph) 默认共享；可强制 0/1 |
| `NCCL_NET_SHARED_COMMS` | 1 | multi-recv 时同 netDev+远端 rank **复用** send/recvComm |
| `NCCL_NET_OPTIONAL_RECV_COMPLETION` | 1 | LL/LL128 单 sub 可跳过严格 recv completion |
| `NCCL_GDRCOPY_SYNC_ENABLE` | 1 | 将 send head 放到 GDRCopy 映射，加速 GPU 读 credit |
| `NCCL_GDRCOPY_FLUSH_ENABLE` | 0 | 额外 PCI 读 flush 路径（与 topo needFlush 配合） |

---

## 6. 生命周期总览

```text
[Topo 选择 NET 边]
    │
    ▼
sendSetup / recvSetup          主线程：选 NIC、GDR、PXN；ProxyMsgSetup
    │                            recv 侧返回 ncclNetHandle_t
    ▼
(bootstrap 交换 connectInfo)
    │
    ▼
sendConnect / recvConnect      主线程：ProxyMsgConnect（可异步 poll）
    │                            Proxy：插件 connect + 分配 map
    │                            主线程：import map → 填 conn.*
    │                            挂上 proxyProgress
    ▼
[运行时每次 collective / p2p]
    Kernel 使用 conn.buffs/head/tail/connFifo
    Proxy 循环: progressOps → send/recvProxyProgress
    │
    ▼
可选: Local/Graph RegisterBuffer → Proxy regMr
    │
    ▼
sendFree / recvFree + *ProxyFree
```

---

## 7. Setup：`sendSetup` / `recvSetup`

### 7.1 `canConnect`（约 161–169 行）

- 默认 `*ret = 1`  
- **同主机**（`hostHash` 相同）时调用 `ncclTopoCheckNet`：可能因策略禁用节点内 NET，改走 SHM/P2P  

### 7.2 `sendSetup`（约 297–336 行）

1. **shared 标志**  
   - 有 `graph`（集体 ring/tree）或 `connIndex==0` → 专用缓冲 `shared=0`  
   - 否则 P2P：由 `NCCL_NET_SHARED_BUFFERS` 决定（默认倾向共享）  

2. **`ncclTopoGetNetDev`**  
   - 选出 `netId` / `netDev` / **`proxyRank`**  
   - **PXN**：`proxyRank != myRank` 时，发送由 **本节点另一 GPU 的 proxy** 出网  
   - `connIndex==0` 且 PXN → `comm->useNetPXN = true`  

3. **`ncclTopoCheckGdr`**  
   - 发送侧是否可用 GDR；置 `NCCL_DIRECT_NIC`  

4. **`ncclProxyConnect(TRANSPORT_NET, 1, proxyRank, …)`**  
   - 连接到 **proxyRank 进程** 的 proxy 服务  

5. **`ncclProxyCallBlocking(..., ProxyMsgSetup, setupReq)`**  
   - 在 proxy 上创建 `sendNetResources`（见 §8）  

6. **connectInfo 输出**  
   - 前部：`topParentRanks[proxyRank]`（给对端 recv 知道发送 proxy 身份）  
   - 后部：`useGdr`  

### 7.3 `recvSetup`（约 344–381 行）

差异要点：

- 选网卡时 **peerRank = 自己**（接收用本 rank 的 NIC）  
- **不支持 PXN on receive**：`proxyRank` 固定为本 rank  
- 若 GDR：`ncclTopoNeedFlush` → `needFlush`  
- Setup 响应 **直接带回 `ncclNetHandle_t`**（listen 句柄），供对端 `connect`  
- 同样附带 `useGdr`  

日志形态：

```text
Channel XX/Y : src[gpu] -> dst[gpu] [send|receive] via NET/<plugin>/<dev>/GDRDMA...
```

PXN 发送日志会多打印 `(proxyRank)`。

---

## 8. Proxy Setup：`*ProxySetup`

### 8.1 `sendProxySetup`（约 733–772 行）

- 分配 `sendNetResources`，填 rank/netDev/GDR/shared/…  
- `getProperties`：  
  - `useDmaBuf = useGdr && dmaBufSupport && PTR_DMABUF`  
  - `maxRecvs`、`netDeviceType/Version`、`maxP2pBytes`（校验合法范围）  
- **不** listen；响应体为空  

### 8.2 `recvProxySetup`（约 774–814 行）

- 同样填 resources  
- **`ncclNet->listen(netDev, respBuff, &netListenComm)`**  
- 响应体 = `ncclNetHandle_t` → 经 bootstrap 到对端 send 的 `connect`  

---

## 9. Connect：主线程与 Proxy 协作

### 9.1 异步协议

`sendConnect` / `recvConnect` 使用 **`ncclProxyCallAsync` + `ncclPollProxyResponse`**：

- 首次进入：分配 `connectMap` 作 `transportResources`，发起 Connect RPC  
- 再次进入（InProgress）：继续 poll  
- 失败且非 InProgress：释放 map  

这与 runtime connect、非阻塞初始化兼容。

### 9.2 `sendProxyConnect`（约 839–1021 行）——最重

**A. 插件 connect**

```text
setNetAttrs
ncclNetGetDeviceHandle(...)

if shared:
  挂 proxyAppend 到 localPeers[tpLocalRank].send
  if maxRecvs>1 && NET_SHARED_COMMS:
    按 netDev×tpRemoteRank 复用 sendComm（仅一个 localRank 真正 connect）
  else:
    ncclNet->connect(...)
else:
  ncclNet->connect(...)  // 专用连接

netSendComm == NULL → done=0, ncclInProgress  // 异步 connect
```

**B. 构造 connectMap 并分配内存**

- 协议缓冲 / sendMem / recvMem 编码进 map  
- DEVMEM：同进程 cudaMalloc 或 cuMem shareable；跨进程 IPC  
- HOSTMEM：同进程 `cudaHostAlloc`；跨进程 `netCreateShm`  
- 可选 GDRCopy `gdcSync`  

**C. 初始化控制面**

```c
head = shared ? -NCCL_STEPS : 0;   // 共享模式先不给 credit
connFifo[i].size = -1;
```

**D. `regMr` / `regMrDmaBuf`**

对每个非空协议缓冲向插件注册，得到 `mhandles[p]`；若有 device net，可 `getDeviceMr`。

**E. 响应**

整份 `connectMap` 拷回主线程。

### 9.3 `recvProxyConnect`（约 1023+ 行）

对称逻辑：

- `accept` / 完成 `netRecvComm`（含 shared comm 复用）  
- 分配 map 与缓冲  
- GDR 时可能分配 `gdcFlush`  
- `regMr` 接收缓冲  
- 返回 map  

### 9.4 主线程 import（`sendConnect` 后半）

- 同进程不同 GPU：可能 `cudaDeviceEnablePeerAccess`  
- 跨进程：`netMapShm`、`ncclP2pImportShareableBuffer`  
- 从 map 解出 GPU 指针，填 `send->conn`（§10）  
- 设置 `proxyProgress = sendProxyProgress`（或 device net 无需 progress 时为 NULL）

---

## 10. GPU 可见的 `ncclConnInfo` 接线

### 10.1 发送侧 conn（GPU 视角）

| 字段 | 指向 | GPU 用途 |
|------|------|----------|
| `head` | sendMem->head 或 gdcSync | 读 credit：proxy 已完成多少 step |
| `tail` | recvMem->tail | 写：发布已写数据 |
| `connFifo` | recvMem->connFifo | 写 size；shared 时读/写 offset |
| `buffs[p]` | 协议缓冲 GPU 指针 | 写 payload |
| `stepSize` | SIMPLE 缓冲 / NCCL_STEPS | 步进 |
| `flags` | DIRECT_NIC 等 | 是否直连 NIC |
| `mhandles` / `netDeviceHandle` | 同进程 device net | 设备侧网络路径 |

### 10.2 接收侧 conn

| 字段 | 指向 | GPU 用途 |
|------|------|----------|
| `head` | sendMem->head | 读：proxy 已 post 到网络多少 |
| `tail` | recvMem->tail 或 gdc | 写：GPU 消费进度 / 同步 |
| `buffs` | 接收缓冲 | 读数据 |
| `connFifo` | 同上 | 与 proxy 协调 size/offset |

**注意方向：** 名字是 sendMem/recvMem，但 **send 连接与 recv 连接交叉映射 head/tail**，形成双向信用。

### 10.3 与 Kernel 的契约（SIMPLE 摘要）

```text
发送 GPU:
  wait (tail - head) < NCCL_STEPS   // 有空 slot
  copy → buff[slot]
  connFifo[slot].size = n
  fence
  tail++

发送 Proxy:
  wait connFifo[slot].size != -1 && tail 足够
  isend(buff, size, mhandle)
  test done
  size = -1; fence; head++

接收 Proxy:
  irecv(buff)
  test done; 可选 flush
  通知 GPU 数据就绪 (tail/size 语义)

接收 GPU:
  wait 数据可见
  读 buff → 用户 recv
  推进 head 等 credit
```

LL/LL128 额外用 **线内 flag** 判断数据完整性（proxy 在 sysmem 路径会扫 flag，见 §12）。

---

## 11. 共享缓冲与共享连接

### 11.1 为何 P2P 要共享缓冲

Ring/Tree 每条 NET 边流量大、生命周期长 → **每连接专用** 大缓冲。  
P2P/sendrecv 边数多、突发 → 专用缓冲显存爆炸 → **`NCCL_SHARED_STEPS` 池 + offset 模式**。

`sharedNetBuffersInit` / `sharedBuffersGet`：

- 按 channel × slot × sub 取偏移  
- `NCCL_SHARED_STEPS = 16` 与 `NCCL_STEPS` 共同限制 pipeline 深度：  
  `maxDepth = min(NCCL_STEPS, NCCL_SHARED_STEPS / nsubs)`

### 11.2 共享连接 `NET_SHARED_COMMS`

当插件 `maxRecvs > 1` 且 env 开启：

- 同一 `netDev` + 同一 `tpRemoteRank` 的多个本地 rank/channel **复用** 一个 `sendComm`/`recvComm`  
- 用 `activeConnect` 选唯一 connector 执行真正的 `connect`  
- `sendRefCount` 管理关闭  

**原理：** 减少 QP/连接数，配合 multi-recv 提高 NIC 效率。

---

## 12. Progress：`sendProxyProgress` 三阶段

状态：`ncclProxyOpReady` → `Progress` → 全部 sub 完成 → `None`。

### 12.1 Ready 初始化

对每个 sub：

```c
sub->base = ROUNDUP(resources->step, chunkSteps);
resources->step = sub->base + nsteps;
posted = transmitted = done = 0;
sendMhandle = mhandles[protocol]  // 非 reg
```

### 12.2 posted：给 GPU 发 credit（共享模式关键）

条件：`posted < nsteps && posted < done + maxDepth`

- **shared + 非 reg**：`sharedBuffersGet` → 写 `connFifo[slot].offset` → fence → 更新 `sendHead`  
- **非 shared**：只推进 `posted` 计数（专用缓冲 credit 在 done 时还）  

GDRCopy：`*gdcSync = ...; wc_store_fence()`。

### 12.3 transmitted：数据就绪则 `isend`

条件：`transmitted < posted && transmitted < done + NCCL_STEPS`

就绪判据：

1. `connFifo[slot].size != -1`  
2. `recvTail > tail`（或 LL 协议放宽）  
3. **LL128 非 GDR**：扫每行 flag 是否等于 `step+1`  
4. **LL**：扫 `flag1/flag2`  
5. **SIMPLE+reg**：`ringAlgo->getNextSendAddr` 取用户缓冲地址与 size  
6. **SIMPLE+shared+reg**：`sendbuff + transmitted * NCCL_MAX_NET_SIZE`  

然后：

```c
setXferNetAttrs(...)
ncclNet->isend(netSendComm, buff, size, tpRank, mhandle, phandle, &request)
// request 非空 → transmitted += sliceSteps
```

### 12.4 done：`test` 完成还 credit

```c
ncclNet->test(request, &done, &size)
if done:
  connFifo[slot].size = -1
  fence                    // 必须先于 head
  if !shared: head = base + done
  done += sliceSteps
```

**顺序铁律：** 先清 `size` 再推 `head`，否则 GPU 可能复用 slot 而 proxy 仍认为 size 有效。

全部 sub `done==nsteps` → op 完成，清理 `ringAlgo` 引用。

---

## 13. Progress：`recvProxyProgress` 多阶段

### 13.1 Ready：按 `recvComm` 分组

- 将相同 `netRecvComm` 的 sub **交换到连续区间**  
- `groupSize ≤ maxRecvs`：一次 `irecv` 收多路  
- 初始化 `base/step`、清零计数、`regBufferReady=0`

### 13.2 posted：组 `irecv`

对组内每个 sub 填 `ptrs/sizes/tags/mhandles`：

| 模式 | 指针来源 |
|------|----------|
| SIMPLE 专用 | `localBuff + slot*stepSize` |
| SIMPLE 共享 | 共享池 offset |
| SIMPLE reg | 等 kernel 启动：`connFifo[base%STEPS].size==-1` 置 ready 后直收用户缓冲 |
| LL/LL128 | 协议 FIFO 槽 |

```c
ignoreCompletion = OPTIONAL_RECV_COMPLETION && (LL|LL128) && subCount==1
irecv(recvComm, subCount, ptrs, sizes, tags, mhandles, phandles, &request)
// 成功则组内各 sub posted += sliceSteps
```

### 13.3 received：`test` 组完成

- 清 `connFifo[slot].size = -1`  
- 若 SIMPLE + GDR + needFlush → 进入 flush  

### 13.4 flush（GDR 可见性）

- **`gdcFlush`**：x86 上 `mfence` + 哑读，强制 PCIe posted write 落 GPU 可见  
- 否则 `ncclNet->iflush`（常为 RDMA Read 刷）  

### 13.5 transmitted / done：通知 GPU 可消费并回收

推进 tail/head 相关计数，使 GPU 原语看到数据；完成后累计 `args->done`。

（细节与 profiler 埋点见源码 1610–1765 行；与 `proxy-internals.md` §5.6 一致。）

---

## 14. GDR / GDRCopy / DMA-BUF / Flush

| 技术 | 在 net.cc 中的位置 | 目的 |
|------|-------------------|------|
| **GDR（DIRECT_NIC）** | setup CheckGdr；缓冲放 DEVMEM；`NCCL_DIRECT_NIC` | NIC DMA 直接碰 GPU 内存 |
| **DMA-BUF** | connect 时 `regMrDmaBuf`；user reg 同样 | 现代驱动注册路径；C2C/PCI 标志 |
| **GDRCopy sync** | `gdcSync` 作 send head | GPU 更快看到 credit |
| **needFlush + iflush/gdcFlush** | recv progress | GPU 读 GDR 写入数据前保证可见性 |
| **双方 useGdr 协商** | connectInfo 中 useGdr；任一方 0 则清 DIRECT_NIC | 避免单侧 GDR 不对称 |

**PCI GDR 模式：** `getHandleForAddressRangeFlags` 在 CUDA≥12.8 且 `useGdr==Pci` 时强制 PCIe DMA-BUF 映射，避免错误走 C2C。

---

## 15. 用户缓冲注册（UB / Graph Register）

### 15.1 入口

| API | 场景 |
|-----|------|
| `ncclNetLocalRegisterBuffer` | 用户 `ncclCommRegister` 等 |
| `ncclNetGraphRegisterBuffer` | CUDA Graph 捕获期注册 + cleanup 队列 |

### 15.2 `netRegisterBuffer` 流程

对每个 peer connector（需 `NCCL_DIRECT_NIC`）：

1. 查 `regRecord->netHandleHead` 是否已对该 `proxyConn` 注册 → 复用 handle  
2. 否则 `ProxyMsgRegister(netRegInfo)` → proxy `regMr`/`regMrDmaBuf`  
3. 挂到 `netHandleHead` 链表  

失败（无 GDR、多段无 DMA-BUF 等）→ `outRegBufFlag=0`，回退 staging 缓冲路径。

### 15.3 Progress 中的 reg 语义

- **Send：** `sub->reg` 时用 `ringAlgo` 或 `sendbuff` 偏移，**不经** 中间 `buffers[SIMPLE]`  
- **Recv：** 等 kernel 表示 ready 后再 `irecv` 直收用户地址，避免与 kernel 启动竞态  

这是大消息 NET 路径的关键优化（零拷贝进 NIC）。

---

## 16. netAttr：给插件的流控提示

### 16.1 `populateCommNetAttrs`

| 连接类型 | maxConcurrentPeers 启发 | maxFlowsPerPeer |
|----------|-------------------------|-----------------|
| P2P only | `p2pnChannels * batch` 与 nRanks 取 min | `p2pnChannelsPerPeer` |
| 集体 | tree arity 相关上限 | `nChannels` |

并设置 op 位图（Send/Recv/AlltoAll… 或集体算法）。

### 16.2 `setXferNetAttrs`

每次 progress 根据 **本 op 的 nPeers/nChannels/algo/proto** 收紧属性，若变化则 `setNetAttr` 通知插件（例如调整 QP/流数量策略）。

**原理：** 插件可按真实并发做内部资源规划，而不是按最坏情况常开。

---

## 17. 设备侧 Net Device Handle

`ncclNetGetDeviceHandle`：

- 当前规则：`NCCL_NET_DEVICE_UNPACK` 且版本匹配且 **recv** 才分配 handle  
- `needsProxyProgress`：若设备路径自推进，可把 `proxyProgress` 置 NULL  

主线程 connect 后把 handle/mhandles 填进 `conn`，供 **设备侧 unpack/网络** 路径使用（与纯 host proxy 路径并存）。

---

## 18. Free / 资源回收

| 函数 | 释放内容 |
|------|----------|
| `sendFree` | 主线程 map；跨 GPU IPC close；跨进程 SHM detach |
| `recvFree` | 主线程 map 指针 |
| `sendProxyFree` / `recvProxyFree` | 插件 closeSend/Recv/Listen；deregMr；cuda/shm free；shared 池引用 |
| `sharedNetBuffersDestroy` | 本地 peer 数组与共享池 |
| Graph cleanup | `cleanupNet` → graph deregister |

Shared comm 路径必须 **refcount** 正确，避免过早 `closeSend`。

---

## 19. 端到端时序图

### 19.1 连接建立（Ring 跨节点边）

```text
Rank A (send)                         Rank B (recv)
     │                                      │
     │ sendSetup → proxySetup               │ recvSetup → listen → handle
     │                                      │
     │◄──────── bootstrap 交换 handle ───────│
     │                                      │
     │ sendConnect async                    │ recvConnect async
     │   proxy: connect(handle)             │   proxy: accept
     │   alloc map, regMr                   │   alloc map, regMr
     │   return map                         │   return map
     │ import map → conn.*                  │ import map → conn.*
     │ proxyProgress = sendProxyProgress    │ proxyProgress = recvProxyProgress
```

### 19.2 一次 SIMPLE 发送切片

```text
GPU A                          Proxy A                         Network / Proxy B / GPU B
 │                                │                                  │
 │ write buff[slot]               │                                  │
 │ size=N; fence; tail++          │                                  │
 │                                │ see size & tail                  │
 │                                │ isend(...)                       │
 │                                │ ───────────────────────────────► │ irecv / RDMA
 │                                │ test done                        │
 │                                │ size=-1; fence; head++           │
 │ read head, reuse slot          │                                  │
```

### 19.3 PXN 发送

```text
GPU0 kernel 写缓冲（可能经 SHM 映射到 Rank1 进程）
Rank1 proxy 线程 isend 出 mlx5（靠近 Rank1 的 NIC）
→ 减少 GPU0 跨 NUMA 敲远端 NIC 的门铃/DMA 路径
```

`sendSetup` 中 `proxyRank != myRank` 即此模式；**recv 仍在本 rank proxy**。

---

## 20. 与 net_ib / CollNet / 其它笔记的关系

| 组件 | 关系 |
|------|------|
| **`net_ib/p2p.cc`** | 实现 `ncclNet->isend/irecv/test`；CTS、QP、resiliency 在此 |
| **`net.cc`** | 调用插件，不感知 IB QP 细节 |
| **`coll_net.cc`** | 平行的集合 offload 适配层 |
| **`proxy.cc`** | 调度 `proxyProgress`、RPC 状态机 |
| **`proxy-internals.md`** | Proxy 框架总览；本文是 NET 适配详解 |
| **`ibv-post-send-stall.md`** | 故障常落在插件 isend 内部；计时应区分 net.cc 与 ibv |
| **注册 / FIFO 专题** | `codex_nccl-chunking-pattern-...md` |

**调试拆分：**

```text
卡在 sendConnect poll          → proxy/RPC/accept
卡在 sendProxyProgress posted  → GPU 不写 tail/size
卡在 transmitted isend NULL    → 插件背压/资源
卡在 test 永不 done            → 网络/对端/CTS（进 net_ib）
```

---

## 21. 阅读与调试建议

### 21.1 阅读顺序（源码）

1. 文件尾 `netTransport` vtable  
2. `sendSetup` / `recvSetup`  
3. `sendProxyConnect` 的 map 分配与 `regMr`  
4. `sendConnect` 如何填 `conn`  
5. `sendProxyProgress` 三阶段  
6. `recvProxyProgress` 分组 + irecv + flush  
7. `netRegisterBuffer` 与 progress 中 `sub->reg` 分支  

### 21.2 推荐日志

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,PROXY,REG
```

关注：`via NET/.../GDRDMA`、`sendConnect`/`Isend posted`、`request done`、register 成功/失败。

### 21.3 关键不变量（复习）

1. **`size=-1` 先于 `head` 更新**  
2. **两端 useGdr 必须协商一致**（否则清 DIRECT_NIC）  
3. **PXN 仅 send**  
4. **shared 模式 head 初值 `-NCCL_STEPS`，靠 posted 逐步放 credit**  
5. **reg 直收前必须等 kernel ready**  
6. **多段缓冲无 DMA-BUF 时 reg 失败属预期**  

---

## 22. 源码索引与修订记录

### 22.1 函数–行号索引（约）

| 主题 | 行号（约） |
|------|------------|
| connectMap 宏与结构 | 25–158 |
| canConnect | 161–169 |
| netAttr | 193–280 |
| sendSetup / recvSetup | 297–381 |
| sendConnect / recvConnect | 431–615 |
| shared buffers | 646–731 |
| send/recv ProxySetup | 733–814 |
| send/recv ProxyConnect | 839–1195 |
| ProxyFree | 1196–1303 |
| sendProxyProgress | 1304–1468 |
| recvProxyProgress | 1470–1765 |
| Register / Dereg | 1767–2083 |
| netTransport vtable | 2085–2092 |

### 22.2 修订记录

| 日期 | 内容 |
|------|------|
| 2026-07-10 | 初稿：`net.cc` 全文件角色、数据结构、setup/connect/progress/reg/GDR/PXN/shared 机制与时序 |

---

## 附录：完整性自检

| 检查项 | 状态 |
|--------|------|
| 文件角色与架构位置 | 有（§1） |
| vtable 全回调覆盖 | 有（§2） |
| 核心数据结构 | 有（§3–4） |
| Setup / Connect / Progress / Free | 有（§7–13、§18） |
| GDR/DMA-BUF/Flush/PXN/Shared | 有（§7、§11、§14、§19） |
| 用户缓冲注册 | 有（§15） |
| 与插件/其它模块边界 | 有（§20） |
| 调试与不变量 | 有（§21） |
| 行号索引便于跳转 | 有（§22） |

讲解按 **生命周期顺序** 组织，进度状态机单独成章，并与现有 `proxy-internals.md` 互补（proxy 框架 vs NET 适配细节）。
