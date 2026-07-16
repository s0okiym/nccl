# NCCL 线程模型总结（v2.30.7）

> 本文基于对 NCCL 2.30.7 源码的逐行核查整理，所有结论附 `文件:行号` 以便验证。
> 内容：NCCL 中所有线程的分配、每个线程的核心任务、运行流程与关键数据结构。

---

## 一、总览：线程的五个作用域

NCCL 的线程按生命周期/作用域分五类。**集合通信的数据面路径上没有任何隐藏线程**——`ncclAllReduce → collectives.cc → enqueue.cc` 全部运行在用户调用线程上（`src/enqueue.cc` 中无任何线程创建）。

| 作用域 | 线程 | 数量 |
|---|---|---|
| 用户线程 | 调用线程本身 + group 异步 job 线程（瞬时） | 每 group ≥2 job 时每 job 1 个 |
| 每进程 | Bootstrap root、RAS、IB async event（每 IB 端口 1 个）、IB port recovery | 各 ≤1 |
| 每 communicator | Proxy service、Proxy UDS、Proxy progress（懒创建）、GIN progress、RMA proxy progress | 各 ≤1 |
| 每连接 | net_socket helper | 每连接 ≤16 |
| GPU device | kernel 内 block/warp 角色 | grid = channel 数，block ≤640 线程 |

辅助事实：

- 所有线程用 `ncclSetThreadName` 命名（`top -H` 可见）：`NCCL BootstrapR`、`NCCL Service %d`、`NCCL UDS Service %d`、`NCCL Progress%d`、`NCCL RAS`、`NCCL IbAsync %d`、`NCCL GIN Progress%d`、`NCCL RMA Proxy Progress%d` 等。
- 线程创建统一走 `std::thread` / `STDTHREADCREATE`（`src/include/checks.h:237`）；只有 vendored/plugin 代码直接用 `pthread_create`。
- `src/os/linux.cc:125` 通过 `pthread_setattr_default_np` 调大默认 pthread 栈。
- 全库**没有 watchdog/heartbeat 线程，也没有线程池**。

---

## 二、Host 线程逐个详述

### 1. 用户调用线程 与 group 异步 job 线程

- 集合调用全程在用户线程：建 `ncclInfo` → 选算法/协议（tuner 表）→ 按 channel 调度 → 上传 work 到 FIFO → `cuLaunchKernelEx` 发 kernel（`src/enqueue.cc:1753-1846`）。首个集合不会懒创建任何线程。
- **Async job 线程**（`src/group.cc:484`，入口 `ncclAsyncJobMain:75`）：`ncclGroupEnd` 内 ≥2 个 job 时每 job 一个线程，用于多 comm 并行 init/destroy/preconnect；单 job 或不在 group 内则内联执行（`group.cc:477`）。非阻塞 group 再开一个总控线程跑 `groupLaunchNonBlocking`（`group.cc:855`）。用完即 join（`group.cc:497`）。

### 2. Bootstrap root 线程 —— 建 comm 的带外聚合服务

- 创建：`bootstrapCreateRoot` → `std::thread(bootstrapRoot)`（`src/bootstrap.cc:419`，入口 `:288`），仅当本进程生成了 uniqueId（`NCCL_COMM_ID` 未设时由 `ncclGetUniqueId` 触发，`src/init.cc:188`）。**detach，永不 join**，收齐信息即退出。
- 流程：监听 socket 先于线程建好（地址随 uniqueId 分发）；accept 循环每连接同步收一个 `extInfo{rank, nranks, listenRootAddress, connectInfo}` 即关闭（`bootstrap.cc:312-372`）——**没有 per-connection handler 线程**；按 rank 序增量缝合 ring：前驱已到则立刻回发，后继已在数组则回赠，否则暂存 `rankInfo[]/rankAddressesRoot[]`，最后统一 flush（`:379-387`）。支持多 root（`nroots>1`，`ncclCommInitRankScalable`）。
- 非 root rank 完全无线程：`bootstrapInit`（`bootstrap.cc:674`）在调用线程上直连 root 报到 → 接受回连拿 next peer → ring 互联 → 沿 ring 做 allgather（`bootstrapAllGather:1194`）交换各 rank 的 proxy/P2P/UDS/RAS 地址。后续 `bootstrapSend/Recv` 也是调用线程上的阻塞 socket 调用，乱序到达的连接停在 `unexpectedConnections` 链表（`:483`）。
- uniqueId 即 `ncclBootstrapHandle{magic, addr, nRanks}`（`src/include/bootstrap.h:14`），magic 用于认证后续每个 socket accept。

### 3. Proxy service 线程 —— proxy 的"控制面"

- 创建：`ncclProxyCreate`（`src/proxy.cc:2081`，入口 `ncclProxyService:1713`）← `initTransportsRank`（`src/init.cc:1529`），comm 初始化时即建；每 `ncclProxyState` 一个（split 子 comm 经 `sharedRes` 共享，refcount，`proxy.cc:2058`）。
- 停止：`ncclProxyStop`（`proxy.cc:2092`）向自己的 listen socket 发 `ncclProxyMsgStop`，并向每个 peer socket 发 `ncclProxyMsgClose`；join 于 `commFree`（`init.cc:285`）。
- 核心任务：**纯 socket 命令循环**（不再兼做数据推进，那是 progress 线程的事）。poll 本地 peer socket 数组 + listen socket（有在途异步 op 时 timeout=0，否则 500ms 以便观察 `abortFlag`，`proxy.cc:1776`）。
- 协议：每条 peer 连接收 4 字节命令 `Init/SharedInit/Setup/Connect/GetFd/Register/Deregister`（`proxyMatchOpType:1692`），封装为 `ncclProxyAsyncOp` 挂到 `ncclProxyLocalPeer.asyncOps`，由 `proxyProgressAsync`（`proxy.cc:1580`）分发到 transport vtable 的 `proxySetup/proxyConnect/proxySharedInit/proxyConnInit/proxyRegister/proxyDeregister`；未完成返回 `ncclInProgress` 下轮重试，完成后回 `ncclProxyRpcResponseHeader{opId,res,respSize}`。
- IB QP 的 INIT/RTR/RTS（`src/transport/net_ib/connect.cc:344/451/469`）就是在该线程上同步完成的——**本版本已没有旧式的 IB 异步建连线程**。
- 客户端侧（用户线程）：`ncclProxyCallAsync / ncclPollProxyResponse / ncclProxyCallBlocking`（`proxy.cc:1339/1366/1425`），乱序响应经 `expectedResponses` 链表按 opId 匹配。

### 4. Proxy UDS service 线程 —— 进程内 fd 传递

- 与 service 线程成对创建（`proxy.cc:2086`，入口 `ncclProxyServiceUDS:1972`），poll 本 rank 的 unix socket（500ms 超时轮询 stop 标志）。
- 只服务两类请求（`ncclIpcHdr` + SCM_RIGHTS）：
  - `ncclProxyMsgGetFd`——把 `CUmemGenericAllocationHandle` 导出为 POSIX fd 回传（`proxyGetFd:1555`，cuMem 跨进程共享）；
  - `ncclProxyMsgQueryFd`——查询远端 rank fd 对应的本地 fd（`proxyQueryFd:1538`，跨 rank buffer 注册）。
- join 于 `commFree`（`init.cc:288`）。

### 5. Proxy progress 线程 —— proxy 的"数据面"（全库最关键的后台线程）

- 懒创建：由 **service 线程**在首个 `tcomm->proxyProgress != NULL` 的连接出现时创建（`proxyConnInit:1527` → `proxyProgressInit:1444` → `std::thread(ncclProxyProgress)`，`proxy.cc:1035`）；每 proxyState 至多一个；退出由 service 线程在收尾时 stop + join（`proxy.cc:1055,1926`）。
- 主循环（`proxy.cc:954-1012`）：反复 `progressOps` 遍历 `state->active` 链表调 `op->progress()`（即 transport 的 `proxyProgress` 函数指针）；空闲或每 `NCCL_PROGRESS_APPENDOP_FREQ`(=8) 轮调 `ncclProxyGetPostedOps` 摄取新 op——**唯一的睡眠点**是无活跃 op 时在 pool 条件变量上 wait（`proxy.cc:854`）；无新增时 `std::this_thread::yield()`。
- Op 生命周期（用户线程 → progress 线程的生产者-消费者链）：
  1. 用户线程 enqueue 时 `ncclProxySaveOp`（先 inquire 后正式，经 `hostStreamPlanCallback` 的 `uploadProxyOps`，`enqueue.cc:1392`）按 pattern 拆成 per-peer `ncclProxyOp`；
  2. `ncclLocalOpAppend`（`proxy.cc:489`）写入 **SysV 共享内存 ops pool**（`/dev/shm/nccl-XXXXXX`，`ncclProxyOpsPool{ops[], nextOps, freeOps[], 进程共享 mutex+cond}`，每 tpLocalRank 独占 `MAX_OPS_PER_PEER` 槽位，`proxy.cc:1457-1467`）；
  3. `ncclProxyStart`（`proxy.cc:1014`）把链挂到 `pool->nextOps` 并 `cond.notify_one()`；
  4. progress 线程批量（默认 16，`NCCL_PROXY_APPEND_BATCH_SIZE`）摄取，`ProxyAppend` 把 `ncclProxyOp` 转成 `ncclProxyArgs` 里的 `ncclProxySubArgs`——shared 连接按 `proxyAppendPtr` 聚合多 local rank 的同 opCount op 为一个 args 多 subs（最多 `NCCL_PROXY_MAX_SUBS = MAXCHANNELS`）；
  5. 状态机 `ncclProxyOpNone/Ready/Progress`；`done == nsubs` 后 `removeOp` 回收。
- 提供 `proxyProgress` 的 transport：
  - **NET** send/recv（`net.cc:1304/1470`：读 connFifo → `isend/irecv` → `test` 轮完成 → 写 head/tail 释放 GPU 槽位）；
  - **COLLNET**（`coll_net.cc:972/1175`：驱动 `iallreduce/iflush` 等）；
  - **P2P** send（仅 `NCCL_P2P_USE_CUDA_MEMCPY` 的 CE-memcpy 模式，`p2p.cc:840`）；
  - **PROFILER** recv（轮询 GPU 写的 `ncclDevProfiler` 计数器，`profiler.cc:23`）。
  - **SHM / P2P / NVLS 正常路径无 proxy progress**——GPU 直接搬数据。
- connFifo 分工：GPU 的 wait 角色线程写 `connFifo[step%8].size`（`src/device/prims_simple.h:119`），proxy 读它取数据并回写 `sendMem->head`；共享 buffer 模式下 proxy 反写 `connFifo[slot].offset` 指引 GPU（`net.cc:1343`）。connFifo 位于 **pinned host 内存**（`ncclNetMap` HOSTMEM bank，`net.cc:957`），GPU 经 PCIe/C2C 访问。

### 6. RAS 线程 —— 进程级可靠性服务

- 每进程仅一个（`rasInitialized` + 互斥 + refcount），`ncclRasCommInit` 于首个 comm 的 bootstrap 时创建（`src/ras/ras.cc:109`，入口 `rasThreadMain:581`），`NCCL_RAS_ENABLE` 可关（默认开）。**comm 销毁不停它**（`ncclRasCommFini` 只清槽位减引用）；进程退出时 `atexit(rasTerminate)` 经通知管道发 `RAS_TERMINATE` 后 join（`ras.cc:164-170`）。
- 单线程 poll 三类 fd：
  1. RAS 网络监听 socket（地址随 bootstrap allgather 分发，进程间互联成 ring 骨干）；
  2. RAS 客户端 socket（`localhost:28028`，`NCCL_RAS_CLIENT_PORT`，服务 `ncclras` CLI 的文本协议 `status/monitor`）；
  3. 本地通知 socketpair `rasNotificationPipe`（`RAS_ADD_RANKS / RAS_TERMINATE`）。
- 消息协议：`CONNINIT/CONNINITACK/KEEPALIVE/PEERSUPDATE/COLLREQ/COLLRESP`（后两者驱动 RAS 广播与聚合集合，如 `RAS_BC_DEADPEER`）。
- **keepalive 不是独立线程**：每轮循环末尾统一跑超时处理器（`src/ras/rasnet.cc`：空闲 1s 发 KEEPALIVE，5s 告警并建 fallback 链路，20s 杀 socket，60s 判 peer 死亡；均可被 `NCCL_RAS_TIMEOUT_FACTOR` 缩放）。

### 7. net_ib 的两个进程级线程

- **IB async event 线程**（`src/transport/net_ib/init.cc:450`，入口 `ncclIbAsyncThreadMain`，`common.cc:110`）：**每物理 IB 端口一个 detached 线程**，阻塞在 `ibv_get_async_event`；只处理异步事件——fatal device/CQ/QP 错误计入 `ncclIbStats`（数据路径经 `ncclIbStatsCheckFatalCount` 同步消费）、端口速率变更；无任务队列，进程退出时随 verbs context 关闭而死，从不 join。
- **IB port recovery 线程**（`p2p_resiliency_recovery.cc:1386`，入口 `:1253`）：每进程一个，需 `NCCL_IB_RESILIENCY_PORT_RECOVERY=1`（默认关）；condvar + `recoveryInbox` 链表接收数据路径上报的故障（`p2p_resiliency.cc:381`），跑 Init→AliveMessages→Ack→Success/Failed 状态机修复失败的 QP（定时参数：启动延迟 200ms、批间隔 500ms、alive 超时 4s、ack 超时 5s、最多 5 次尝试）；`ncclIbFinalize` 时停止并 join。
- net_ib 其余动作（GDR flush 的 RDMA read、resiliency 探测/重放、QP 重传）全部内联在 proxy progress 线程或数据路径上完成，**无额外线程**。

### 8. net_socket helper 线程

- 仅 socket 传输启用：每 send/recv 连接 `NCCL_SOCKET_NTHREADS`（≤`MAX_THREADS=16`）个 `persistentSocketThread`（`src/transport/net_socket.cc:579`，入口 `:290`），首个任务时懒创建，各自 condvar 任务队列驱动 `nSocks/nThreads` 条 TCP 并行收发；连接关闭时 join（`:773-781`）。

### 9. GIN progress 线程 与 RMA proxy progress 线程

- **GIN**（`src/gin/gin_host.cc:297`，入口 `ncclGinProgress:43`）：每 comm ≤1，仅当某 GIN context 报 `needsProxyProgress`（PROXY 后端恒真，GDAKI 仅当 QP 标了 cpu_proxy）。状态字 `1=run / 0=sleep / -1=exit` + mutex/cond；循环对每个 ctx 调 `ginProgress()`——PROXY 后端即轮询 GPU 写入的 GFD 描述符队列（host pinned 内存，CI/PI 索引），向 RMA 插件发 `iput/iget/iflush` 并 test 完成。`commFree` 时 join（`init.cc:329`）。
- **RMA proxy**（`src/rma/rma_proxy.cc:454`，入口 `ncclRmaProxyProgressThread:337`）：每 comm 一个，首个 RMA window 注册时创建（`ncclRmaProxyConnectOnce`）。状态机多一个 `2=reclaim 暂停握手`；推进逻辑（`rma_proxy_progress.cc:269`）：完成在途 desc、观察到 GPU 写的 `readySeq` 后从 per-peer 环形缓冲发新 desc、轮询持久化 graph desc、调插件 `rmaProgress`。`commFree` 时 join（`init.cc:330`）。

### 10. 插件线程（参考实现，非核心库）

- inspector profiler 的 dump 线程（`plugins/profiler/inspector/inspector.cc:610`，定期导出 JSON/Prometheus）。
- example CE profiler 的 poller（`plugins/profiler/example/profiler_plugin_ce.cc:242`，CE 事件归因）。
- vendored DOCA 库里有个 `priv_service_mainloop` 线程（`gdaki/doca-gpunetio/src/doca_gpunetio.cpp:1097`），但 NCCL 从不创建它（无调用者，实际为死代码）。

---

## 三、GPU device 线程组织

### 1. 发射几何：grid/block ↔ channel

- `ncclLaunchKernel`（`enqueue.cc:1753`）：**`gridDim.x = popcount(plan->channelMask)`——一个 CUDA block 负责一个 channel**；`blockDim.x = plan->threadPerBlock`（取 plan 内各 task `nWarps*32` 的最大值，下限 `NCCL_MIN_NTHREADS=128`）。驱动 API `cuLaunchKernelEx`（sm90+ 可带 CGA cluster 等属性）。
- 线程数规则（`enqueue.cc:2099-2116` + `graph/tuning.cc:243-257`）：基础值来自 tuner 表 `comm->maxThreads[algo][proto]`；RING+SIMPLE 额外 **+1 个 sync warp**，TREE+SIMPLE +4 warp，TREE 强制 `NCCL_MAX_NTHREADS=640`；p2p kernel 恒 640；环境变量 `NCCL_NTHREADS`（SIMPLE/LL，上限 512）与 `NCCL_LL128_NTHREADS`（上限 640）可调。
- kernel 内（`ncclKernelMain`，`src/device/common.h:355`）：block n 认领 channelMask 第 n 个置位对应的 channel（`common.h:366-373`）；**warp 0 装载 `ncclKernelComm`、warp 1 装载 `ncclDevChannel`、warp 2+ 装载 work batch 到共享内存**，`__syncthreads` 后进入主循环；batch 按 blockIdx 轮转分配（block b 得 batch b, b+nChannels, …）；单个 coll work 仅 `tid < work->nWarps*32` 的线程参与。

### 2. warp/线程角色分工（协议相关）

- **SIMPLE**（`src/device/prims_simple.h:605-626`）：按组内 tid 区间定角色——`tid<nrecv` 为 `RoleWaitRecv`，之后 `RoleWaitSend`，**最后一个 warp** 为 `RolePostSend/RolePostRecv`；搬数据只用 `nworkers = nthreads − 1个sync warp`（nthreads≥96 时）。Wait 角色自旋等步计数并**写 connFifo.size 通知 proxy**；Post 角色 `fence_acq_rel_sys` 后**发布 conn->head/tail**——两者都是不搬数据的专职同步线程。
- **LL / LL128**（`prims_ll.h` / `prims_ll128.h`）：无角色标志，全线程搬数据；`tid < fan.nsend()` 的线程兼管 send 同步与 `sendConnFifo`，**最后 warp 的 lane** 兼管 recv credit 回写（`*recvConnHeadPtr = ++recvConnHead`）。
- **NVLS / PAT / net-offload kernel**：整个 block（恒 640 线程）按 tid 区间切成 scatter/gather/reduce/bcast 等职能 warp 组（`src/device/all_reduce.h:391-408`），组间用 named barrier（`barrier_sync(15-group)`）同步。
- **sendrecv kernel**：warp 0 先用 ballot 算好工作划分，warps 均匀分到各 work，再各自分裂为 send warp / recv warp（`src/device/sendrecv.h:106-174`）。

### 3. 两条关键 FIFO（host↔device、device↔proxy）

- **Work FIFO**（host→GPU）：`comm->workFifoBuf`（pinned 或 GDR 映射，`NCCL_GDR_COPY_FIFO`；confidential computing 下禁用）。三种上传模式（`uploadWork`，`enqueue.cc:1248`）：Args（嵌进 4KiB kernel 参数 `ncclDevKernelArgs4K`）、Fifo（memcpy 进环形缓冲，`workFifoProduced` 推进）、Persistent（graph 捕获用 `cudaMallocAsync`）。GPU 侧 `loadWorkBatchToShmem` 每线程 16B 并行装载；`workFifoConsumed` 由 kernel 完成后的 cudaEvent 回调更新（`enqueue.cc:1862-1898`）。
- **connFifo**（GPU↔proxy）：`ncclConnFifo{mode,offset,size,ptr}` × `NCCL_STEPS=8`，嵌在 pinned host 内存的 `ncclRecvMem`（`comm.h:67-77`）。语义：**GPU wait 线程写 size → proxy 读 size/offset 取数 → proxy 写 head 释放槽位；共享 buffer 时 proxy 反写 offset 供 GPU 读**（`net.cc:1331-1448`）。recv 方向 proxy 完成 `irecv` 后置 `size=-1` 哨兵，GPU 等它复用已注册的用户 buffer。

### 4. `nccl_device` 公开 API 的线程模型

- 一切 device 集合以 **coop**（协作线程组）为模板参数：`ncclCoopThread / Warp / Tile<N> / Lanes / WarpSpan / Cta`（`src/include/nccl_device/coop.h`），同步原语分别为 `__syncwarp` / named barrier（`__barrier_sync_count(1+id, 32*nWarps)`）/ `__syncthreads`。
- **GIN 网络操作由 coop 内 `thread_rank()==0` 的 leader 线程单独发起**（固定模式 `coop.sync(); if (leader) issue; coop.sync();`，见 `impl/gin__funcs.h:467-493`）；LSA barrier 以 team（world/lsa/rail）为作用域 arrive/wait（`lsa_barrier.h`）。
- 对称集合 kernel 同样 grid=channel 数、CTA 级 coop、warp 跨步循环，TMA 路径由 **lane 0 选举**发 `cp.async.bulk`（`src/device/symmetric/all_gather.cuh`）。
- `src/devcomm/` shim 按**编译时的 NCCL 版本**（2.29.2 / 2.29.7 / 2.30.0 三档）快照 device ABI（`ncclDevComm` 等布局的 offset/size static_assert），保证旧 header 编译的用户 kernel 兼容新 libnccl——与 CUDA driver 版本无关。

---

## 四、一次跨节点 AllReduce 的线程协作时序

1. **用户线程**：`ncclAllReduce` → 规划（tuner 选 ring/tree + LL/LL128/SIMPLE）→ work 写 FIFO → 发 kernel 到用户 stream → 经 `cudaLaunchHostFunc` 回调（`hostStreamPlanCallback`）把 proxy ops 写入 SHM pool 并 notify → 返回（异步）。
2. **GPU**：每 channel 一个 block 跑 `ncclDevKernel`；节点内段由 primitives 直接经 NVLink/P2P 搬运；网络段由 wait/post 角色线程写 connFifo/head。
3. **Proxy progress 线程**：被 cond 唤醒 → 摄取 op → `send/recvProxyProgress` 读 connFifo、发 `isend/irecv`（IB 即 `ibv_post_send`）、`test` 轮 CQ → 回写 head/tail 释放 GPU 槽位，直至 `done == nsubs`。
4. **Proxy service 线程**：稳态下基本空闲，只在建连/注册/销毁时跑 RPC；**UDS 线程**仅在 cuMem fd 传递时工作；**RAS 线程**全程独立做 keepalive/状态上报。
5. 完成：kernel 结束 → cudaEvent 回调更新 `workFifoConsumed` → 用户 `cudaStreamSynchronize` 拿到结果。

---

## 五、关键数据结构速查

| 结构 | 位置 | 作用 |
|---|---|---|
| `ncclProxyState` | `src/include/proxy.h:333` | 每 comm 的 proxy 总状态（三线程、socket、client 侧 peer 数组、progressState） |
| `ncclProxyOpsPool` | `proxy.h:229` | SHM 中的 op 池（进程共享 mutex+cond，per-rank 槽位切片） |
| `ncclProxyOp` → `ncclProxySubArgs` / `ncclProxyArgs` | `proxy.h:73/140/185` | op 从共享池槽位到活跃链表的转化与聚合 |
| `ncclProxyConnection` | `proxy.h:398` | 连接状态机 + transport vtable 指针 + proxyAppendPtr |
| `ncclTransportComm` | `src/include/transport.h:117` | transport vtable（setup/connect/free + 7 个 proxy 钩子） |
| `ncclConnFifo` / `ncclSendMem` / `ncclRecvMem` | `src/include/collectives.h:72` / `src/include/comm.h:53-77` | GPU↔proxy 连接 FIFO 与 head/tail |
| `ncclIbDev` / `ncclIbNetCommBase` / `ncclIbSendComm` / `ncclIbRecvComm` | `src/transport/net_ib/common.h:79/340/429/508` | IB 设备、连接基类（reqs/qps）、CTS FIFO 与 flush QP |
| `ncclShmem` / work batch | `src/device/common.h:48/140` | block 共享内存态与 work 批装载 |
| `ncclGinState` / `ncclRmaProxyState` | `src/include/gin/gin_host.h:38` / `src/include/rma/rma_proxy.h:185` | GIN / RMA 各自 progress 线程的状态 |

---

## 六、一句话概括

NCCL 的 CPU 侧是"每 comm 一个 proxy 三线程组（service 控制面 + UDS fd 服务 + progress 数据面）+ 每进程若干全局服务线程（bootstrap root、RAS、IB event/recovery）+ GIN/RMA 两个按需 progress 线程"；GPU 侧是"一 channel 一 block、组内按 tid 区间分工（搬运 warp + 专职同步/post 线程）"，两侧经 pinned 内存里的 work FIFO 与 connFifo 两条环形队列解耦。
