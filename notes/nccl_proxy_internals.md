# NCCL Proxy 子系统与 Progress 线程完全解析（v2.30.7）

> 本文逐行基于 `src/proxy.cc`（2151 行）、`src/include/proxy.h`（469 行）及 `src/enqueue.cc`、`src/transport/net.cc` 相关段落整理，所有结论附 `文件:行号`。
> 前置阅读：`notes/nccl_thread_model.md`（线程全景）。

---

## 0. 为什么需要 proxy

GPU kernel 能自己搬 NVLink/PCIe/SHM 上的数据，但有两类事它做不了：

1. **驱动网络**：`ibv_post_send`、socket send/recv、CQ 轮询等 verbs/插件调用只能在 CPU 上执行；
2. **跨进程协调**：cuMem 句柄导出 fd、跨进程共享内存注册等。

NCCL 为此在每个 communicator 上挂一个 **proxy 子系统**，由三条线程组成（`src/proxy.cc`）：

| 线程 | 入口 | 角色 |
|---|---|---|
| Proxy service | `ncclProxyService`（`proxy.cc:1713`） | 控制面：poll socket，执行建连/注册等 RPC |
| Proxy UDS service | `ncclProxyServiceUDS`（`proxy.cc:1972`） | 进程内 fd 传递（cuMem/QueryFd） |
| Proxy progress | `ncclProxyProgress`（`proxy.cc:954`） | 数据面：驱动 GPU 做不了的传输（net send/recv 等） |

主线程（用户调用线程）是 proxy 的**客户端**：通过 socket RPC 让 service 线程建连，通过共享内存 op 池让 progress 线程推数据。两条生产者-消费者通道：

```
主线程 ──TCP socket RPC──▶ service 线程   （建连/注册，一次性，慢路径）
主线程 ──SHM op 池───────▶ progress 线程  （每集合每次传输，热路径）
主线程 ──UDS────────────▶ UDS 线程       （cuMem fd 传递，极少）
```

---

## 1. 核心数据结构全解

### 1.1 `struct ncclProxyState`（`proxy.h:333-382`）

每个 top-parent communicator 一个（split 子 comm 经 `comm->sharedRes->proxyState` 共享，`refCount` 计数）。字段按用途分组：

- **身份/配置**：`refCount`、`comm`、`tpRank/tpnRanks`（本 rank 在 top-parent comm 中的 rank/总数）、`tpLocalnRanks`（本节点 rank 数）、`cudaDev`、`p2pnChannels/p2pChunkSize`、`buffSizes[NCCL_NUM_PROTOCOLS]`、`dmaBufSupport`、`directMode`。
- **后端句柄**：`ncclNet`、`ncclCollNet`、`ginState`、`netContext/collNetContext`、`netAttr`、`profilerContext`、`memManager`、`abortFlag`（指向 `comm->abortFlag`，本地多 rank 共享的 uint32）。
- **三条线程与监听者**：`thread`（service）、`threadUDS`、`listenSock`（TCP，`bootstrapInit` 时创建）、`ipcSock`（UDS 监听）、`stop`、`asyncResult`（progress 线程记录的首个错误，release 序写入，`proxy.cc:982`）。
- **客户端侧（主线程用）**：`peerAddresses`（每个 tpRank 的 proxy TCP 地址，bootstrap allgather 得来）、`peerSocks`（每个 tpRank 一条 socket）、`proxyOps`（每个 tpRank 一份 SHM 池映射 + 本地追加游标 `ncclProxyOps`）、`sharedDevMems`、`peerArraySize`、`peerIpcSock`、`peerAddressesUDS`。
- **progress 侧**：`progressState`（见 1.5）。
- **RPC 完成队列**：`expectedResponses`（见 4.3）。

### 1.2 连接：`ncclProxyConnection` 与连接池

`ncclProxyConnection`（`proxy.h:398-412`）描述一条"某个 transport 的某条 send/recv 连接在 proxy 侧的句柄"：

- `send`（1=send 连接）、`transport`（0..4：P2P/SHM/NET/COLLNET/PROFILER）、`shared`（共享连接，见 6.5 聚合）；
- `tpLocalRank`、`sameProcess`、`sock`（回包用的客户端 socket）、`tcomm`（指向 transport vtable，见 1.3）；
- `proxyAppend` / `proxyAppendPtr`：指向该连接 op 链**尾部** args 的指针，供 O(1) 追加与聚合（见 6.5）；
- `transportResources`：transport 私有资源（如 NET 的 `sendNetResources`）；
- `state`：连接状态机 `connUninitialized → connInitialized → connSharedInitialized → connSetupDone → connConnected`（`proxy.h:384-391`；各迁移点见 4.2）；
- `proxyMemHandleQueue`：proxy 侧注册的内存句柄队列。

**服务端连接池** `ncclProxyConnectionPool`（`proxy.cc:1069-1097`）：按 128 个一组（`NCCL_PROXY_CONN_POOL_SIZE_POW2=7`）分 bank 动态扩容；连接 id = `bank<<7 | offset`。池内存不回收复用，service 线程退出时统一 `ncclProxyFreeConnections`（逐连接调 transport 的 `proxyFree`，`proxy.cc:1112-1125`）。

关键设计：`ncclProxyInitResp.connection` 把**服务端连接结构体的裸指针**回给客户端（`proxyConnInit`，`proxy.cc:1522`），之后客户端每次 RPC 都把这个指针原样发回，服务端直接解引用——它本质是一个跨 socket 的不透明句柄（仅在同一进程内有效，但 NCCL 的 proxy 连接语义保证客户端与该 service 同进程或同节点）。

### 1.3 transport vtable：`ncclTransportComm`（`transport.h:117-134`）

```c
struct ncclTransportComm {
  setup / connect / free;                                   // 主线程侧
  proxySharedInit / proxySetup / proxyConnect / proxyFree;  // service 线程侧（RPC 驱动）
  proxyProgress;                                            // progress 线程侧（热路径）
  proxyRegister / proxyDeregister;                          // service 线程侧（buffer 注册）
};
```

实例在 `ncclTransports[]`（`transport.cc:15-18`）。各 transport 的 proxy 钩子：

| transport | proxySetup/Connect | proxyProgress | 说明 |
|---|---|---|---|
| P2P | 有 | 仅 CE-memcpy 模式（`p2p.cc:1478`） | 正常路径 GPU 直写对端显存 |
| SHM | 有 | **无**（`shm.cc:470`） | 只需要 proxy 做建连映射 |
| NET | 有 | send/recv（`net.cc:1304/1470`） | proxy 的主力消费者 |
| COLLNET | 有 | send/recv（`coll_net.cc:972/1175`） | |
| NVLS | 无 | 无 | 全 GPU |
| PROFILER | recv 侧 | recv（`profiler.cc:23`） | 伪 transport，轮询 GPU 计数器 |

`proxyProgress == NULL` 是"该连接不需要 progress 线程"的判据，贯穿建连（决定是否懒建线程）与 op 保存（`SaveProxy` 直接跳过，`proxy.cc:589`）。

### 1.4 op 的两种形态：`ncclProxyOp` → `ncclProxyArgs`/`ncclProxySubArgs`

**`ncclProxyOp`**（`proxy.h:73-131`）是主线程写入 SHM 池的**定长槽位消息**。要点：

- `next` 是 **int 型池内索引而非指针**（`proxy.h:78`）——SHM 段在不同进程映射地址不同，指针不可移植；
- 携带一次传输的全部上下文：`nbytes/nsteps/chunkSize/sliceSteps/chunkSteps/channelId/dtype/redOp/coll/pattern/protocol/algorithm`、`sendMhandle/recvMhandle`（注册句柄）、`sendbuff/recvbuff`、`ringAlgo`、`specifics`（collnetDirect/bcast 附加参数）、profiler 相关字段；
- `opCount`：全局序号，编码见 6.3；
- `enqNext`：挂在 plan 的 `proxyOpQueue` 时用（enqueue 阶段）。

**`ncclProxyArgs`**（`proxy.h:185-220`）是 progress 线程里的**运行时 op 实例**：

- `subs[NCCL_PROXY_MAX_SUBS]`（= `MAXCHANNELS` = 64，`proxy.h:58`）：聚合后的子操作数组（见 6.5）；
- `progress`：函数指针，取自 `connection->tcomm->proxyProgress`（`proxy.cc:431`）；
- `state`：`ncclProxyOpNone/Ready/Progress`（`proxy.h:49-53`）；
- `done`（已完成 sub 数）、`idle`（本轮有无进展，供调度）；
- 三级链接：`next`（active 链表）、`nextPeer`（同连接的后继 op 链）、`proxyAppendPtr`（指回连接的尾指针槽）。

**`ncclProxySubArgs`**（`proxy.h:140-183`）是每个连接/通道切片的进度状态，核心是一组**步进计数器**：

- `base`：本 op 在连接步序号空间中的起点（对齐 `chunkSteps`，progress 函数 Ready 阶段算好）；
- `posted`：已向 GPU/网络提供的缓冲数；
- `received`：网络已收到的步数（recv 用）；
- `flushed`：已 flush 的步数（GDR recv 用）；
- `transmitted`：已发给网络（send）/ 已通知 GPU 数据就位（recv）的步数；
- `done`：彻底完成（槽位已释放）的步数；
- `requests[NCCL_STEPS]`：每个缓冲槽的在途网络请求句柄；
- `groupSize`：recv 时共享同一 `recvComm` 的连续 sub 数（multi-recv 聚合，`net.cc:1473-1510`）。

### 1.5 SHM op 池：`ncclProxyOpsPool` / `ncclProxyOps`

```c
struct ncclProxyOpsPool {                       // proxy.h:229
  struct ncclProxyOp ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS];
  volatile int nextOps, nextOpsEnd;             // 已发布链（消费者取）
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];   // 每本地 rank 一条空闲链
  std::mutex mutex; std::condition_variable cond;
};
```

- 尺寸：`MAX_OPS_PER_PEER = 2 * MAXCHANNELS * 2 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH = 2*64*2*8 = 2048`（`proxy.h:227`；注释：要容下所有 channel 两轮 op，每 p2p work 含 send+recv 两个 op 故乘 2）；`NCCL_MAX_LOCAL_RANKS = 72`（`device.h:92`）。
- 位于 `/dev/shm/nccl-XXXXXX`（`proxyProgressInit`，`proxy.cc:1457`）；`ncclOsSetMutexCondShared` 把 mutex/cond 设为**进程共享**（`proxy.cc:1469`）。
- **每个本地 rank 独占一段** `[r*MAX_OPS_PER_PEER, (r+1)*MAX_OPS_PER_PEER)`：初始化时 `freeOps[r]` 串好该段空闲链（`proxy.cc:1462-1467`）。这样生产者（各 rank 主线程）几乎无竞争。
- 为什么放共享内存：**op 的生产者（主线程）和消费者（progress 线程）可能不在同一进程**——PXN 发送时 op 要投给本节点另一 rank 的 proxy（见 3.2）；同进程时则只是统一的快速通道。

`ncclProxyOps`（`proxy.h:238-245`）是主线程侧对某个对端 proxy 池的**客户端句柄**：`pool`（映射地址）、`handle`（shm 句柄）、`count`、`freeOp`（本地缓存的下一个空闲槽，避免每次都原子操作）、`nextOps/nextOpsEnd`（未发布的本地链）。

### 1.6 `ncclProxyProgressState`（`proxy.h:272-286`）

```c
struct ncclProxyProgressState {
  struct ncclProxyOpsPool* opsPool;   // 自家 SHM 池（消费者视图）
  ncclShmHandle_t handle; char opsPoolShmSuffix[16];
  std::thread thread; volatile int stop;
  struct ncclProxyPeer** localPeers;                 // shared 连接的 per-local-rank 聚合状态
  struct ncclSharedNetComms* netComms[NCCL_MAX_NETDEVS]; // 共享 net comm（refcount per channel）
  struct ncclProxyArgs *active, *pool;               // 活跃 op 链 / 空闲 args 链
  struct ncclProxyPool* pools;                       // args 的内存块（每块 NCCL_MAX_OPS=2048 个）
  int nextOps;                                       // 上次批量摄取的剩余
};
```

### 1.7 `ncclConnFifo` 与 head/tail（GPU↔proxy 协议）

```c
struct ncclConnFifo { int mode; int offset; ssize_t size; void* ptr; };  // collectives.h:72
```

- `NCCL_STEPS = 8` 个槽（`device.h:26`），数组嵌在 `ncclRecvMem`（另有 `tail`）；`ncclSendMem` 持有 `head`（`comm.h:53-77`）。
- NET transport 把它们分配在 **pinned host 内存**（`ncclNetMap` 的 HOSTMEM bank，`net.cc:957`），GPU 经 PCIe/C2C 读写，proxy 线程直接访问。
- 协议（send 方向为例）：
  - GPU 填完一个缓冲槽 → 写 `connFifo[step%8].size = 字节数`（`prims_simple.h:119`），并推进 `recvMem->tail`；
  - proxy 读 `size`/`offset` → `isend` → `test` 完成 → 先复位 `size = -1`、内存屏障、再写 `sendMem->head` 释放槽位（`net.cc:1436-1449`）；
  - 共享缓冲模式：proxy 反写 `connFifo[slot].offset` 告诉 GPU 用哪块共享缓冲（`net.cc:1343`）；
  - GDC 模式（GDRCopy）：head/tail 改放 CUDA 显存（`gdcSync`），写后需 `wc_store_fence()` 冲掉 WC 写缓冲（`net.cc:1346-1349`）。

---

## 2. 生命周期

### 2.1 创建

1. `bootstrapInit` 时 `ncclProxyInit`（`proxy.cc:2038`；`bootstrap.cc:860`）：`new ncclProxyState`，挂上已建好的 `listenSock`，初始化 UDS 监听 `ipcSock`。此时尚无线程。
2. `initTransportsRank` 时 `ncclProxyCreate`（`proxy.cc:2054`；`init.cc:1529`）：仅当 `refCount==1` 时填充配置字段并创建 **service 线程**（`proxy.cc:2081`）与 **UDS 线程**（`proxy.cc:2086`）。split 子 comm 只加引用不建线程。
3. **progress 线程懒创建**：首个 `tcomm->proxyProgress != NULL` 的连接初始化时，由 **service 线程**（在 `proxyConnInit` 里，`proxy.cc:1527-1528`）调 `proxyProgressInit`（建 SHM 池）→ `ncclProxyProgressCreate`（`proxy.cc:1032`）创建。由 service 线程创建的好处：CPU 亲和性（`NCCL_PROXY_CPUSET`）和 CUDA context 直接继承（注释 `proxy.cc:956`）。

### 2.2 建连（transport connect 路径）

以 NET send 为例（`net.cc:297-336`），主线程在 `initTransportsRank` 的 connect 阶段：

1. `ncclTopoGetNetDev` 选出网卡和 **proxyRank**——PXN 场景下可以是本节点另一个 rank（`net.cc:310-316`；`proxyRank != myRank` 时置 `comm->useNetPXN`）。recv 恒用本 rank 自己的 proxy（`net.cc:368`，"We don't support PXN on receive yet"）。
2. `ncclProxyConnect`（`proxy.cc:1142`）：按 tpRank 取/建一条到该 rank proxy 的 TCP socket（`peerSocks[tpRank]`，懒初始化 `peerSocks/proxyOps/sharedDevMems` 数组），发 `ncclProxyMsgInit` RPC（阻塞），拿回服务端 `ncclProxyConnection*` 与 SHM 池后缀；若该 transport 需要 progress，则 `ncclShmOpen` **映射对方的 op 池**（`proxy.cc:1215-1220`）。
3. `ncclProxyCallBlocking(ncclProxyMsgSetup, ...)`：service 线程执行 transport 的 `proxySetup`（分配缓冲、建 verbs 保护域等）。
4. 对端 rank 完成对称的 setup 后，主线程交换 `ncclConnect` 信息（bootstrapSend/Recv），再 `ncclProxyCallBlocking(ncclProxyMsgConnect, ...)`：service 线程执行 `proxyConnect`（IB 侧在此做 QP INIT/RTR/RTS，`net_ib/connect.cc:344/451/469`）。

Register/Deregister（用户 buffer 注册）同样走 RPC（`ncclProxyMsgRegister/Deregister`）。

### 2.3 停止与销毁

1. `ncclProxyStop`（`proxy.cc:2092`；`commDestroySync` 调用，`init.cc:2762`）：引用归零时——
   a. 未 abort 则自建 socket 向自己的 listenSock 发 `ncclProxyMsgStop`（`proxy.cc:2098-2106`）；
   b. 遍历 `peerSocks`：关 SHM 映射、`cudaIpcCloseMemHandle`、发 `ncclProxyMsgClose`、关 socket；
   c. `stop = 1`（release 序），通知 UDS 线程退出。
2. **service 线程收尾**（`proxy.cc:1925-1935`，顺序重要）：
   `ncclProxyProgressDestroy`（置 `progressState.stop=1` + cond 唤醒 + **join progress 线程** + 释放 args 池）→ 关所有 peer socket → `ncclProxyFreeConnections`（逐连接 transport `proxyFree`）→ 关 listenSock → `proxyOpsFree`（关 SHM）。先停 progress 再释放连接资源，避免 progress 函数访问已释放的 transport 资源。
3. `commFree`（`init.cc:284-289`）：join service 与 UDS 线程。
4. `ncclProxyShmUnlink`（`init.cc:1708`）：所有本地 rank 映射完后解除 SHM 文件名；`ncclProxyDestroy`（`proxy.cc:2137`）：释放数组、`expectedResponses`，`delete proxyState`。

abort 语义：`abortFlag` 置位后，service 循环转入 `PROXY_ABORT`（`proxy.cc:1773`），**但只要还有 peer 连接就必须活着**（`proxy.cc:1769-1772` 注释：等对端也 abort 并关闭，否则对方可能 segfault）；progress 循环则立即退出（`proxy.cc:1009-1010`），在途 op 被丢弃。

---

## 3. Service 线程逐段解析（`ncclProxyService`，`proxy.cc:1713-1944`）

### 3.1 主循环结构

- 启动：应用 `NCCL_PROXY_CPUSET` 亲和性（`std::call_once`，`proxy.cc:1716-1717`）、`cudaSetDevice`。
- poll 数组：`maxProxyConnections = max(tpnRanks+1, NCCL_MAX_LOCAL_RANKS+1=73)`（`proxy.cc:1731`；注释：支持跨 clique P2P 的大量 peer），最后一格固定为 `listenSock`（`proxy.cc:1753`）。
- **超时策略**：`timeout = asyncOpCount ? 0 : 500`（`proxy.cc:1776`）——有未完成 RPC 时非阻塞轮询（驱动 `ncclInProgress` 重试），否则 500ms 上限以便及时观察 `abortFlag`（注释 `proxy.cc:1774`：永不许在 poll 里无限阻塞）。
- 循环条件：`stop == PROXY_RUNNING || npeers > 0`（`proxy.cc:1769`）。
- listenSock 可读 → `ncclSocketAccept` 到空闲槽，`npeers++`（`proxy.cc:1811-1833`）。
- 每个 peer 槽位两轮处理（`proxy.cc:1835-1922`）：
  1. **先推进该 peer 所有 pending `asyncOps`**：`proxyProgressAsync` 返回 `ncclInProgress` 的留到下轮；返回真错误则关连接；
  2. **再读新命令**：`ncclSocketTryRecv` 取 4 字节 type——`Stop`（置 stop + 关连接）/ `Close`（关连接）/ `proxyMatchOpType` 命中的七类（Init/SharedInit/Setup/Connect/GetFd/Register/Deregister）→ `proxyServiceInitOp`；未知命令告警并关连接。

### 3.2 RPC 协议帧与 `proxyProgressAsync`

请求帧（客户端 `ncclProxyCallAsync` 发送，`proxy.cc:1349-1356`）：

```
type(int) | connection(void*) | reqSize(int) | respSize(int) | reqBuff(reqSize) | opId(void*)
```

服务端 `proxyServiceInitOp`（`proxy.cc:1656-1690`）逐段读取、分配 `respBuff`、挂入 `peer->asyncOps` 链表、`asyncOpCount++`，并**立即尝试执行一次** `proxyProgressAsync`。

`proxyProgressAsync`（`proxy.cc:1580-1654`）按 type 分发到 vtable，并推进连接状态机：

| type | 调用 | done 后状态迁移 |
|---|---|---|
| `Init` | `proxyConnInit`（建连接池条目；若需 progress 则建池建线程） | → `connInitialized`（`proxy.cc:1534`） |
| `SharedInit` | `tcomm->proxySharedInit` | → `connSharedInitialized`（`:1600`） |
| `Setup` | `tcomm->proxySetup` | → `connSetupDone`（`:1624`） |
| `Connect` | `tcomm->proxyConnect` | → `connConnected`（`:1627`） |
| `Register` | `tcomm->proxyRegister` | — |
| `Deregister` | `tcomm->proxyDeregister` | — |

（`GetFd` 虽在 `proxyMatchOpType` 白名单中，但 `proxyProgressAsync` 无对应分支会报错——真实 GetFd 走 UDS，见第 5 节。）

`done=0`（如 IB 连接需要多次推进）→ 返回 `ncclInProgress`，op 留在队列下轮继续。`done=1` → 回包：

```
ncclProxyRpcResponseHeader{opId, res, respSize} | respBuff(respSize)   （proxy.cc:1635-1643）
```

注释（`proxy.cc:1630-1633`）：setup/connect 完成后即使后续出错也不能再断开连接——respBuff 可能已发给请求方，断连会让对方 segfault。

### 3.3 客户端 API 与乱序响应

- `ncclProxyCallAsync`（`proxy.cc:1339`）：发请求帧后把 `opId` 挂入 `expectedResponses` 链表（预分配 respBuff）。
- `ncclPollProxyResponse`（`proxy.cc:1366`）：先查链表；未命中则非阻塞读 socket——收到的是**别的 op 的响应**就存入链表（`expectedProxyResponseStore`）返回 `ncclInProgress`，是自己 op 的则取走（`expectedProxyResponseRemove`）。因为同一 socket 上多个异步 RPC 的响应按**完成顺序**而非请求顺序到达，必须有这个乱序缓存。
- `ncclProxyCallBlocking`（`proxy.cc:1425`）：`opId = malloc(1)`（仅作唯一句柄），Async + 自旋 Poll 直至完成。

---

## 4. UDS 线程（`ncclProxyServiceUDS`，`proxy.cc:1972-2036`）

- 单 fd poll（`ipcSock`），500ms 超时轮询 `stop`/`abortFlag`。
- 请求格式 `ncclIpcHdr{type, rank, reqSize, respSize, opId, data[16]}`（`proxy.h:324-331`，128 字节内联数据 + SCM_RIGHTS 传 fd）：
  - `ncclProxyMsgGetFd`：把本 rank 的 `CUmemGenericAllocationHandle` 经 `cuMemExportToShareableHandle` 导出为 POSIX fd，经 UDS 回传给同节点请求方（`proxyGetFd`，`proxy.cc:1555-1578`）。客户端：`ncclProxyClientGetFdBlocking`（`proxy.cc:1281`，用 `comm->gproxyConn[proxyRank]`——为此时连一条到对端 proxy 的 TRANSPORT_P2P 连接）。
  - `ncclProxyMsgQueryFd`：跨 rank buffer 注册时查询远端 fd 对应的本地 fd（`proxyQueryFd`，`proxy.cc:1538-1552`；批量版 `ncclProxyClientBatchQueryFdBlocking`）。
- 每次请求-响应都是一次性 UDS 连接（`ncclProxyCallBlockingUDS`，`proxy.cc:1228-1277`：以 opId 哈希作为临时地址）。

---

## 5. Progress 线程详解（`ncclProxyProgress`，`proxy.cc:954-1012`）——本文核心

### 5.1 主循环逐行

```c
do {
  int idle = 1;
  ret = progressOps(proxyState, state, state->active, &idle);   // ① 推进所有活跃 op
  if (ret != ncclSuccess) { 记录 asyncResult; break; }
  // ② idle 沿变化时给 profiler 记 Idle/Active 事件
  if (idle || !state->active || (++proxyOpAppendCounter == NCCL_PROGRESS_APPENDOP_FREQ)) {
    proxyOpAppendCounter = 0;
    ret = ncclProxyGetPostedOps(proxyState, &added);            // ③ 摄取新 op
    if (added == 0) std::this_thread::yield();                  // ④ 无活则让出 CPU
  }
  lastIdle = idle;
} while ((stop==0 || (stop==1 && state->active)) && !abortFlag); // ⑤ 排空后退出
```

要点：

- ① `progressOps`（`proxy.cc:801-831`）：遍历 `active` 链表调 `op->progress(proxyState, op)`；`op->idle` 由 progress 函数自己置位（本轮是否有实际动作）；`state == ncclProxyOpNone` 或出错 → `removeOp`。
- ③ **摄取节流**（`proxy.cc:972-977` 注释）：每次循环都调 `ncclProxyGetPostedOps` 会让小消息性能回退，故只在 (a) 本轮 idle、或 (b) 无活跃 op、或 (c) 距上次摄取已满 `NCCL_PROGRESS_APPENDOP_FREQ`（默认 8）轮时才摄取。
- ⑤ 停止语义：`stop=1` 后**排空 active 链表才退出**；`abortFlag` 则立即退出。

### 5.2 op 生产链：从一次集合调用到 SHM 池

完整链路（以一次跨节点 allreduce 为例）：

```
ncclAllReduce
└─ enqueue.cc 规划阶段
   └─ ncclAddProxyOpIfNeeded（enqueue.cc:108）
      └─ ncclProxySaveOp(comm, op, &needed)   ← justInquire 模式：只判断要不要 proxy
         └─ 若需要：堆拷贝 op 挂入 plan->channels[c].proxyOpQueue
└─ kernel 发射后，strong stream 上的 cudaLaunchHostFunc 回调
   └─ hostStreamPlanCallback（enqueue.cc:1452）
      └─ hostStreamPlanTask（enqueue.cc:1436）
         ├─ uploadProxyOps（enqueue.cc:1392）   ← 重编号 opCount，正式 ncclProxySaveOp
         └─ ncclProxyStart（proxy.cc:1014）     ← 发布到池并 notify
```

**opCount 编码**（`enqueue.cc:1407-1423`）：最低位是 p2p 标记——集合 op：`opCount = (comm->collOpCount << 1) + plan内序号`；p2p op：`(channel 的 p2pOpCount << 1) + 奇数序号`。保存后恢复原值以便 persistent plan（graph 捕获）重放。opCount 是 6.5 节聚合的键。

**`ncclProxySaveOp` 的 pattern 分解**（`proxy.cc:602-766`）：把"一次集合"按通信模式拆成 per-peer 的收发 op——

| pattern | SaveProxy 调用 |
|---|---|
| Ring/RingTwice/Pipeline | recv←`ring->prev`、send→`ring->next`（connIndex 0）；`NeedProxy` 判定（ring 恒真；pipeline 链端 rank 不需要）；AllGatherV 用 `specifics.bcast` 的 slices 修正 nsteps |
| TreeUp/Down/UpDown | 上行：recv←`tree->down[0..arity)`、send→`tree->up`；下行镜像 |
| CollnetChain/Direct | send/recv `up`/`out`，send 用 connIndex 1 |
| Nvls / NvlsTree | `nvls.out`（1/0）；NvlsTree 六条（treeUp/treeDown[1]/treeDown[2] 各收发） |
| PatUp/PatDown | 先完整跑一遍 `PatRSAlgorithm/PatAGAlgorithm` 数出每维度步数，再对 `rank±2^i` 各收发 |
| Send/Recv（p2p） | `root==comm->rank` 跳过；否则对 `root` 存一条（connIndex 1） |
| Profiler | `SaveProxyProfiler`：op 指向 `comm->profiler.workStarted/workCompleted` 计数器 |

**`SaveProxy`**（`proxy.cc:578-598`）：取 `channel->peers[peer]->send/recv[connIndex]` 连接器；`proxyConn.proxyProgress == NULL`（如 SHM）直接跳过——**不需要 progress 的 transport 根本不会进池**。

**`ncclLocalOpAppend`**（`proxy.cc:489-555`）——生产者写入池的精确动作：

1. 用 `proxyConn->tpLocalRank` 选中目标 proxy 的 `ncclProxyOps`（可能是自己的，也可能是 PXN 对端的池映射）；
2. **取空闲槽**：优先用本地缓存 `proxyOps->freeOp`；耗尽则对 `pool->freeOps[本rank]` 做 `atomic_exchange(-1, acquire)` 取自旋等待（`std::this_thread::yield`，`proxy.cc:506-510`）——取到的是该 rank 独占段的链头；
3. `memcpy` 整个 op 进槽，`op->next = -1`，补 `op->connection`，挂到本地未发布链 `proxyOps->nextOps` 尾；
4. **满 `MAX_OPS_PER_PEER`(2048) 的截链保护**（`proxy.cc:526-552`）：提前发布一个前缀——但必须截在 **opCount 边界**（不回切最后一个 opCount 的 op），注释明确：同一 opCount 的 op 拆到不同批次会破坏 progress 线程按 opCount 聚合 subs。

**`ncclProxyStart` / `ncclProxyPost`**（`proxy.cc:1014/477`）：对每个有待发 op 的 peer，持 `pool->mutex` 把本地链接到 `pool->nextOpsEnd`；若池空则置 `nextOps` 并 `cond.notify_one()` 唤醒 progress 线程；最后 `comm->opCount++`。

### 5.3 消费：`ncclProxyGetPostedOps`（`proxy.cc:835-916`）逐段拆解

这是 progress 线程从 SHM 池摄取新 op 的**唯一入口**，也是 `ncclProxyOp`（池槽位消息）与 `ncclProxyArgs`（运行时实例）两种形态之间的"搬运工 + 回收站"。理解它的前提是 op 在池里的三条组织不变式（详见 1.5）：

1. **槽位按生产 rank 切片**：rank r 独占 `ops[r*MAX_OPS_PER_PEER, (r+1)*MAX_OPS_PER_PEER)`，故 `opIndex / MAX_OPS_PER_PEER` 直接解码出生产者身份，无需任何元数据（`proxy.cc:876`）；
2. **已发布链只有一条**：所有生产者的 op 经 `ncclProxyPost`（持 `mutex`）串到同一条 `pool->nextOps` 链，链内可混杂不同 rank、不同 opCount 的 op；
3. **空闲链按 rank 分**：消费完的槽位必须归还到**原生产 rank** 的 `freeOps[i]`，否则切片布局被破坏。

函数体分四段：

#### 段 1：续处理检查（`proxy.cc:840`）

```c
if (state->nextOps != -1) goto process_nextops;
```

`state->nextOps`（progress 线程私有）是**上一轮批量处理没消化完的剩余链头**。若上轮因批量上限截断，本轮跳过一切加锁逻辑直接续跑——截断后续处理不碰 `mutex`。

#### 段 2：取链——三种锁策略（`proxy.cc:845-861`）

```c
std::unique_lock<std::mutex> lock(pool->mutex, std::defer_lock);
if (state->active != NULL && (pool->nextOps == -1 || !lock.try_lock()))
  return ncclSuccess;                       // 情形 A

if (state->active == NULL) {                // 情形 B
  lock.lock();
  if (pool->nextOps == -1 && !state->stop) {
    ... ProxyCtrlSleep 事件 ...
    pool->cond.wait(lock);                  // 全线程唯一睡眠点
    ... ProxyCtrlWakeup 事件 ...
  }
}
state->nextOps = pool->nextOps;             // 情形 C：摘下整条链
pool->nextOps = pool->nextOpsEnd = -1;
```

| 情形 | 条件 | 行为 | 设计理由 |
|---|---|---|---|
| A | `active != NULL` 且（池空 **或** `try_lock` 失败） | 直接返回，一个 op 也不取 | 热路径优先：手里有活就干，绝不为取新 op 阻塞；`try_lock` 失败说明生产者正在发布，下轮再来 |
| B | `active == NULL` 且池空 | `cond.wait` 睡眠 | 彻底空闲才允许睡眠，这是整条线程**唯一**的阻塞点（profiler 记 Sleep/Wakeup 事件） |
| C | 拿到锁且池非空 | 整条链摘下，池头尾置 -1，解锁 | 批量转移所有权，持锁时间最短 |

两个并发正确性要点：

- **无丢失唤醒**：`cond.wait` 前在锁内复查 `pool->nextOps == -1`；生产者 `ncclProxyPost` 在同一把锁内"池空才 `notify_one`"（`proxy.cc:477-487`）——经典条件变量模式，不会睡死。
- **情形 A 中无锁读 `pool->nextOps` 是良性竞争**：最坏结果是这轮漏取，下轮补上；它不参与睡眠判定，不影响唤醒正确性。

#### 段 3：批量消费循环（`proxy.cc:867-895`）

```c
for (int opIndex = state->nextOps; opIndex != -1;) {
  struct ncclProxyOp* peerOp = pool->ops + opIndex;
  int peer = opIndex / MAX_OPS_PER_PEER;                    // 槽位索引 → 生产 rank
  if ((lastOpCount && peerOp->opCount != lastOpCount) ||    // 组边界：opCount 变了
      ((lastPeer != -1) && peer != lastPeer)) count++;      //         或 rank 变了
  if (count == ncclParamProxyAppendBatchSize() + 1) break;  // 批量上限（默认 16 组）
  lastOpCount = peerOp->opCount; lastPeer = peer;
  if (peerOp->connection == NULL) return ncclInternalError; // 防御性检查
  if (peerOp->next != -1) COMPILER_PREFETCH(pool->ops + peerOp->next);
  NCCLCHECK(ProxyAppend(state, peerOp));                    // ★ 消息 → 运行时形态
  (*added)++;
  int lastOpIndex = opIndex;
  opIndex = peerOp->next;
  // 就地用 next 字段构建"待归还"链（倒挂到 freeOp[peer] 头上）
  if (freeOp[peer] == -1) freeOpEnd[peer] = lastOpIndex;
  else                    peerOp->next = freeOp[peer];
  freeOp[peer] = lastOpIndex;
  state->nextOps = opIndex;                                 // 每步更新续处理指针
}
```

要点：

- **批量上限按"组"计而不是按 op 个数计**：一组 = 同 rank 且同 opCount 的**连续** op 段，逻辑上就是"某 rank 某次集合通信的一批收发 op"。整组摄取保证 `ProxyAppend` 能完整地做 subs 聚合（见 5.4），不会把同一聚合单元切成两半。这正是生产者端 `ncclLocalOpAppend` 满 `MAX_OPS_PER_PEER` 时宁可提前截链发布、也必须在 opCount 边界截断（`proxy.cc:526-552`）的原因——两端约定一致。
- **`break` 时不消费当前 op**：`state->nextOps` 在上一轮迭代末尾已指向它，剩余链原样留下轮（对应段 1 的 goto）。
- **`COMPILER_PREFETCH`**：沿 `next` 索引预取下一个槽位，SHM 上指针追逐的标准优化。
- **回收链是倒序就地构建的**：直接复用被消费槽位的 `next` 字段，把槽位串到本 peer 的私有链 `freeOp[peer]`（头）/ `freeOpEnd[peer]`（尾）上——零额外内存。
- **`ProxyAppend` 是唯一的形态转换点**：转换完成后，`ncclProxyOp` 槽位就没有存在价值了，立即进入待回收链。

#### 段 4：无锁归还（`proxy.cc:897-911`）

```c
for (int i = 0; i < proxyState->tpLocalnRanks; i++) {
  if (freeOp[i] == -1) continue;
  int oldFree = -1, newFree = freeOp[i];
  oldFree = COMPILER_ATOMIC_LOAD(&pool->freeOps[i], acquire);
  do {
    pool->ops[freeOpEnd[i]].next = oldFree;       // 链尾接到当前空闲链头前
  } while (!COMPILER_ATOMIC_COMPARE_EXCHANGE(&pool->freeOps[i], &oldFree, newFree,
                                             /*success=*/release, /*failure=*/acquire));
}
```

- 每个 rank 的回收链用**一次 CAS** 整体 prepend 回 `pool->freeOps[i]`——注意全程**不持 `mutex`**，槽位回收是无锁的。
- CAS 失败说明有生产者并发地 `atomic_exchange(-1)` 抢走了空闲链（`ncclLocalOpAppend`，`proxy.cc:506-510`）：`oldFree` 被 CAS 自动更新为新值，重挂链尾重试，直到成功。
- **内存序**：成功 release / 失败 acquire。`freeOpEnd[i].next = oldFree` 的写先于 release-CAS 发布，生产者 acquire-exchange 拿到链头后沿链读到的每个 `next` 都是已初始化的。
- 归还后槽位重新进入生产者的可用集：生产者先消耗本地缓存 `proxyOps->freeOp`，耗尽才原子 exchange 抢整条空闲链（`proxy.cc:497-514`）——**生产者批量拿、消费者批量还**，原子操作频率都摊薄到 1/链长。

#### 槽位流转闭环与两级空闲链

一个槽位的一生：

```
[空闲] pool->freeOps[r] ──atomic_exchange──▶ 生产者本地缓存 proxyOps->freeOp
[写入] 主线程 memcpy op 内容 + 填 connection → 挂本地未发布链 proxyOps->nextOps
[发布] ncclProxyPost：持 mutex 接到 pool->nextOps（池空则 notify_one）
[摄取] GetPostedOps：摘链 → 按 (peer,opCount) 分组批量处理
[转换] ProxyAppend：ncclProxyOp ──拷贝──▶ ncclProxySubArgs（装入 ncclProxyArgs）
        ├─ 同 shared 连接同 opCount → 聚合为 sub
        ├─ 同连接新 opCount        → 挂 nextPeer
        └─ 连接空闲                → 挂 state->active 尾
[推进] progressOps 每轮调 args->progress()（如 sendProxyProgress 三阶段）
[完成] state → ncclProxyOpNone → removeOp：
        ├─ args 结构体 → 归还 state->pool 空闲链（堆上，allocateArgs 复用）
        └─ ncclProxyOp 槽位 → 段 4 CAS 归还 pool->freeOps[r]
```

注意这里有**两级独立的空闲链**，别混淆：

- **op 槽位级**（SHM，`pool->freeOps[r]`）：本函数段 4 负责归还，生产者跨进程复用；
- **args 实例级**（progress 线程堆内，`state->pool` + 分块的 `ncclProxyPool` 内存 bank，每块 `NCCL_MAX_OPS=2048` 个）：`allocateArgs`/`removeOp` 负责，纯线程内复用。

#### 设计取舍小结

1. **摘链式摄取**（锁内只做头指针交换）：持锁时间 O(1)，生产者几乎永远遇不到锁竞争，所以生产者发布可以无脑持锁。
2. **有活不取、空才睡眠**：`active != NULL` 时 `try_lock` 失败即返——网络收发延迟以微秒计，为取新 op 阻塞哪怕一次都会反映到带宽上；彻底空闲时 `cond.wait` 避免烧 CPU。
3. **按 (peer, opCount) 分组限量**：批量上限（`NCCL_PROXY_APPEND_BATCH_SIZE=16`）以逻辑聚合单元计，既限制单次摄取时延，又保证聚合完整性；配合主循环里的 `NCCL_PROGRESS_APPENDOP_FREQ=8` 节流（`proxy.cc:993`），小消息场景下摄取开销被压到最低。
4. **槽位索引即元数据**：`opIndex / MAX_OPS_PER_PEER` 同时给出生产者身份和归还目的地，跨进程零额外协议。
5. **回收无锁化**：归还是消费者侧唯一与生产者并发写的点，用 CAS 重试解决，不进 `mutex` 临界区。

实际观察手段：profiler 插件里 `ProxyCtrlSleep/Wakeup/Append/AppendEnd` 四个事件正好对应段 2 和段 3 的边界；`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROXY` 能看到发布/摄取日志。

### 5.4 `ProxyAppend`：三级聚合结构（`proxy.cc:438-475`）

每条连接有自己的 op 流水，`active` 链表只挂"队首"：

```
state->active ──▶ argsA ──next──▶ argsB ──▶ ...
                    │nextPeer        （每个 args 属于一条连接）
                    ▼
                  argsA'（同连接、更晚的 opCount）
```

对到来的 op（带其 `connection`）：

1. `*connection->proxyAppendPtr` 非空（该连接已有 op 在跑）：
   - **`shared` 连接且 opCount 相同** → `ncclProxyOpToArgs(op, args, args->nsubs)` 聚合为一个新 **sub**（多 local rank 共享同一 NIC 连接时，把同一次集合的各家切片合并执行；subs 一致性校验：sliceSteps/chunkSteps/protocol/dtype/redOp/coll 必须相同且 args 仍处 Ready，`proxy.cc:399-412`）；
   - 否则 → 新 args 挂为尾部 args 的 `nextPeer`，`proxyAppendPtr` 指向新尾。
2. 为空（该连接空闲）→ 新 args 挂到 `active` 链表尾，建立 `proxyAppendPtr`。

args 内存来自 `allocateArgs`（`proxy.cc:208-232`）：按 `PROXYARGS_ALLOCATE_SIZE = NCCL_MAX_OPS = 2048` 一块块 `ncclCalloc`，空闲链 `state->pool` 复用。

**`removeOp`**（`proxy.cc:768-799`）：op 完成（`state==None`）时——若有 `nextPeer`，**用 nextPeer 原位顶替** active 链表中的位置（保持同连接 FIFO 序，迭代指针同步修正）；否则清 `*proxyAppendPtr` 并摘除；归还空闲链。

### 5.5 一个 progress 函数内部：NET send（`sendProxyProgress`，`net.cc:1304-1468`）

这是"GPU 与 proxy 接力"的标准范式，每个 sub 三阶段流水：

```
posted  ──▶  transmitted  ──▶  done
(提供缓冲)    (发向网络)        (释放槽位)
```

- **Ready 初始化**：`sub->base = ROUNDUP(resources->step, chunkSteps)`（连接级步序号对齐），`resources->step = base + nsteps` 留给下一个 op；计数器清零；取协议对应的 mhandle；`args->state → Progress`（`net.cc:1306-1318`）。
- **posted 阶段**（`net.cc:1335-1356`）：`posted < nsteps && posted < done + maxDepth`（`maxDepth = min(NCCL_STEPS, NCCL_SHARED_STEPS/nsubs)`）时——共享缓冲模式：经 `sharedBuffersGet` 取一块共享缓冲，把偏移写入 `connFifo[buffSlot].offset`（seq_cst 屏障），并**提前推进** `sendMem->head = base+posted-NCCL_STEPS` 告诉 GPU"这些槽可填"；非共享只推 `posted += sliceSteps`。
- **transmitted 阶段**（`net.cc:1358-1430`）：发现 `connFifo[buffSlot].size != -1 && recvMem->tail > tail`（GPU 已填好）——LL128 无 GDR 时要逐行校验 128B 行旗标、LL 校验 `flag1/flag2`（数据在系统内存时 GPU 只保证 threadfence）；SIMPLE+reg 经 `ringAlgo->getNextSendAddr` 取注册的用户 buffer——然后 `ncclNet->isend()`，成功则 `transmitted += sliceSteps`。
- **done 阶段**（`net.cc:1432-1457`）：`test(request)` 完成 → **先** `connFifo[buffSlot].size = -1` + 屏障、**再**写 `sendMem->head = base+done` 释放槽位给 GPU（顺序不能反）；`done == nsteps` → `args->done++`，`ringAlgo` 引用计数清理。
- `args->done == nsubs` → `args->state = ncclProxyOpNone`，progress 线程下轮 `removeOp`。

### 5.6 NET recv（`recvProxyProgress`，`net.cc:1470-1765`）五阶段

- Ready：把共享同一 `recvComm` 的 subs 排成连续组（`groupSize`，一次 `irecv` 收多个 sub 的数据，`net.cc:1473-1510`）；
- **posted**：组装 multi-recv 数组（ptrs/sizes/tags/mhandles）；SIMPLE+shared 写 `connFifo` offset；**reg 直收用户 buffer 前等 kernel 启动**（`connFifo[base%8].size == -1` 才置 `regBufferReady`，`net.cc:1546`）；`irecv(subCount,...)`；LL/LL128 单 sub 可用 `NCCL_NET_OPTIONAL_RECV_COMPLETION` 省一次 test；
- **received**：`test` 整组完成 → 每 sub `connFifo[slot].size = -1`、`received += sliceSteps`；GDR 且需 flush 时进入 flush（`gdcFlush`：x86 上 `mfence` + 一次显存哑读强制 PCIe posted write 落盘，`net.cc:1644-1659`；否则 `iflush` 发 RDMA read）；
- **transmitted**：flush 请求完成 → 写 `recvMem->tail = base+transmitted`（或 gdcSync）——**此刻 GPU 才可见数据**；
- **done**：等 GPU 侧 `sendMem->head > base+done`（GPU 已消费）→ 调插件 `irecvConsumed` 回收 multi-recv 请求 → `done += sliceSteps`；
- 全部 sub done → `state = None`。

### 5.7 其他 transport 的 proxyProgress

- **P2P**（CE-memcpy 模式，`NCCL_P2P_USE_CUDA_MEMCPY`）：`p2pSendProxyProgress`（`p2p.cc:840`）发 `cudaMemcpyAsync` D2D + `cudaEventRecord`，轮询 `cudaEventQuery`；
- **COLLNET**：`sendProxyProgress/recvProxyProgress`（`coll_net.cc:972/1175`）驱动 `iallreduce/iflush` 等；
- **PROFILER**：`profilerProxyProgress`（`profiler.cc:23`）轮询 GPU 写入的 `ncclDevProfiler.workStarted/workCompleted` 计数器，为 kernel/channel 事件打时间戳；它复用 sub 的计数器字段做别样用途（注释 `profiler.cc:19-22`）。

---

## 6. 并发与内存模型要点

- **op 池是无锁 + 小段锁的混合**：每 rank 独占槽位段，生产者平时只动本地游标；槽位回收（progress→生产者）用 `freeOps[i]` 的原子交换/CAS；发布/摄取用 `pool->mutex + cond`（进程共享）。
- **`ncclProxyOp.next` 是 int 索引**：SHM 跨进程映射基址不同，不能用指针。
- **连接状态机全部 `atomic_store(release)`**（`proxy.cc:1534/1600/1624/1627`），供主线程与 service 线程跨线程观察。
- **connFifo/head/tail 是 GPU↔CPU 的跨设备协议**：proxy 侧 `volatile` 读 pinned 内存；顺序敏感处用 `std::atomic_thread_fence(seq_cst)`（如先清 size 再写 head）；GDC 路径写显存后 `wc_store_fence()`。
- **两个 stop 字段别混淆**：`proxyState->stop`（service/UDS 线程）vs `progressState.stop`（progress 线程）；abort 另有共享的 `abortFlag`。
- **opCount 即同步点**：shared 连接跨 rank 聚合 subs 依赖所有 rank 对同一集合并发出相同的 opCount；`ncclLocalOpAppend` 截链、`uploadProxyOps` 重编号都为此服务。

---

## 7. 可调参数与调试手段

| 旋钮 | 默认 | 作用 |
|---|---|---|
| `NCCL_PROXY_APPEND_BATCH_SIZE` | 16 | progress 线程每轮摄取 op 的批量上限（`proxy.cc:833`） |
| `NCCL_PROGRESS_APPENDOP_FREQ` | 8 | 活跃期每多少轮才摄取一次新 op（`proxy.cc:926`） |
| `NCCL_PROXY_CPUSET` | — | service/UDS 线程 CPU 亲和性（progress 线程继承 service 的，`proxy.cc:930-952`） |
| `NCCL_PROXY_DUMP_SIGNAL` | -1 | 设为 SIGUSR1/2 后，向进程发信号即打印 active op 全状态（`proxy.cc:920-925`） |
| `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=PROXY` | — | proxy 全链路日志（建连、op 收发、状态迁移） |

**hang 排查**：`NCCL_PROXY_DUMP_SIGNAL=10` 后 `kill -10 <pid>`，`dumpProxyState`（`proxy.cc:291-367`）打印每个活跃 op 的 per-sub 状态字母——recv：`I`(nit)/`R`(eceiving)/`F`(lushing)/`G`(PU wait)/`D`(one)；send：`I`/`G`(PU wait)/`S`(ending)/`D`（`proxy.cc:261-290`），可直接定位卡在 GPU 未填数、网络未完还是 flush。`DEBUG_PROXY` 编译宏（`proxy.cc:234`）可打印每次 append/remove。

---

## 8. 一次跨节点 AllReduce 的 proxy 全链路时序

```
[主线程]                          [GPU]                        [proxy progress]              [对端 proxy progress]
ncclAllReduce
 ├─ 规划 + ncclProxySaveOp(inquire)
 │   → op 入 plan->proxyOpQueue
 ├─ 发射 kernel
 └─ cudaLaunchHostFunc 回调:
     uploadProxyOps (重编号 opCount)
     ncclProxySaveOp(正式) ─┐
     ncclProxyStart ────────┼─▶ SHM 池 nextOps + notify ──▶ cond 唤醒
                            │                                ├─ GetPostedOps 批量摄取
                            │                                ├─ ProxyAppend → active 链
                            ▼                                ▼
                    kernel 运行: wait 角色线程           sendProxyProgress:
                    填缓冲 → connFifo[i].size ─────────▶ posted/transmitted:
                    → recvMem->tail                        test/isend ─────────────────▶ recvProxyProgress:
                                                                                            irecv→test→flush
                    ◀── sendMem->head (槽位释放) ◀── done: size=-1, head=base+done          → tail=... (通知 GPU)
GPU 收端消费 → sendMem->head ─────────────────────────────────────────────────────▶ recv done: irecvConsumed
                            │                                ▼
                            └─ 所有 sub done → state=None → removeOp → 槽位 CAS 归还 freeOps
kernel 结束 → cudaEvent 回调更新 workFifoConsumed → 用户 stream 同步返回
```

service 线程在此过程中全程不参与数据传输——它只在建连、注册、销毁时工作；UDS 线程仅在 cuMem fd 传递时工作。

---

## 9. FAQ（设计取舍）

- **Q：为什么 service 和 progress 分成两个线程？** 旧设计是单线程两状态（opsStart/progress）轮转；现版本把 socket RPC（控制面，低频、可阻塞 poll）与数据推进（热路径、忙轮询 GPU 旗标）拆开，避免 RPC 处理延迟卡住网络收发。
- **Q：为什么 progress 线程由 service 线程创建？** 懒创建（只有用到需要 progress 的 transport 才建），且直接继承 service 已设好的 CUDA context 与 CPU 亲和性（`proxy.cc:956` 注释）。
- **Q：为什么 op 池放 /dev/shm 而不是进程内堆？** 生产者可能是另一进程的本节点 rank（PXN 发送、同节点 cuMem/directMode 场景）；共享内存 + int 索引 + 进程共享 mutex/cond 是统一解。
- **Q：为什么 poll 超时固定 500ms？** 注释（`proxy.cc:1774`）：service 线程绝不可无限阻塞，否则看不到 abortFlag；500ms 是响应 abort 与 CPU 空转的折中。UDS 线程同理（500ms）。
- **Q：为什么 `ncclLocalOpAppend` 满了要在 opCount 边界截链提前发布？** 同一 opCount 的 op 若被拆进两个批次，progress 线程会建成两个 args 而无法聚合成 subs（`proxy.cc:527-529` 注释）。
- **Q：主线程与 progress 线程谁写 connFifo？** 都不写——connFifo 是 GPU kernel（wait 角色线程）与 progress 线程之间的协议；主线程只参与建连与 op 投递。
- **Q：`ncclProxyMsgStart/Abort` 怎么没人发？** 枚举保留（`proxy.h:436-437`），当前版本启停走 `ncclProxyStart`（直写池）与 MsgStop/MsgClose/abortFlag，不在 matchOpType 白名单内。

---

## 10. 关键文件索引

| 文件 | 内容 |
|---|---|
| `src/proxy.cc` | 三线程主体、RPC、op 池、SaveOp/Append/Progress 框架 |
| `src/include/proxy.h` | 全部数据结构定义（带注释） |
| `src/include/transport.h:117-134` | transport vtable |
| `src/enqueue.cc:108-117, 1392-1460` | op 的 inquire/上传/发布链路 |
| `src/transport/net.cc:1304-1765` | NET send/recv progress 状态机范本 |
| `src/transport/net.cc:297-381` | proxyRank/PXN 选择与 Setup RPC |
| `src/device/prims_simple.h:100-173` | GPU 侧 connFifo/head/tail 的另一半协议 |
