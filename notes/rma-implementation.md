# NCCL RMA 实现机制与原理

## 1. 概述

NCCL 的 RMA（Remote Memory Access）功能是一套基于**对称内存窗口（symmetric window）**的单边通信原语，提供三个核心用户 API：

- `ncclPutSignal`：将本地数据写入对端窗口，并可附带一个 signal。
- `ncclSignal`：仅向对端发送 signal。
- `ncclWaitSignal`：等待一个或多个 peer 的 signal 累计到达指定次数。

RMA 的核心设计是**双路径执行**：

- **CE 路径**：对端 rank 与当前 rank 同属一个 LSA（Local System Address）team，走 GPU Copy Engine 直接做 D2D 拷贝。
- **Proxy 路径**：对端不可通过本地 GPU 直连访问，走 RMA 网络插件（IB/GIN 等），由独立 CPU Proxy 线程推进网络操作。

`ncclWaitSignal` 的 peer 列表会被拆分到两条路径并行处理。

---

## 2. 用户 API

公开头文件来源：`src/nccl.h.in`

```c
ncclResult_t ncclPutSignal(
    const void* localbuff, size_t count, ncclDataType_t datatype,
    int peer, ncclWindow_t peerWin, size_t peerWinOffset,
    int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclSignal(
    int peer, int sigIdx, int ctx, unsigned int flags,
    ncclComm_t comm, cudaStream_t stream);

ncclResult_t ncclWaitSignal(
    int nDesc, ncclWaitSignalDesc_t* signalDescs,
    ncclComm_t comm, cudaStream_t stream);
```

当前实现限制：

- `sigIdx` 必须为 0。
- `flags` 必须为 0。
- 数据窗口需先通过 `ncclCommWindowRegister` 注册为 symmetric window。

---

## 3. 任务生命周期

### 3.1 API 入口 → Task

- `src/collectives.cc` 将 API 包装为 `ncclInfo`，调用 `ncclEnqueueCheck`。
- `src/enqueue.cc:3020` 在 `taskAppend` 中识别 RMA，进入 `rmaTaskAppend`。
- `rmaTaskAppend` 进行参数校验，并将大 `ncclPutSignal`（>1GB）拆成多个 `ncclTaskRma` chunk，仅最后一个 chunk 发 signal。
- 每个任务挂入 `planner->rmaTaskQueues[ctx]`，按 RMA context 分队列。

`ncclTaskRma` 结构定义：`src/include/comm.h:277`

### 3.2 Group End → Plan

`ncclGroupEnd` 期间调用 `scheduleRmaTasksToPlan`（`src/rma/rma.cc:138`）：

- 取第一个非空 ctx 队列。
- 按函数类型生成一个 `ncclKernelPlan`。
- 对 Put/Signal 任务，批量合并同一 ctx 内的连续任务。
- 根据 LSA 可达性将任务分到 `plan->rmaTaskQueueProxy` 或 `plan->rmaTaskQueueCe`。

### 3.3 Plan Launch

`src/group.cc:358`：

```c
if (plan->isCeColl) {
  ncclLaunchCeColl(comm, plan);
} else if (plan->isRma) {
  ncclLaunchRma(comm, plan);   // src/rma/rma.cc:90
} else {
  ncclLaunchKernel(comm, plan);
}
```

`ncclLaunchRma` 根据 `func` 分派到 `ncclRmaPut` 或 `ncclRmaWaitSignal`。若 plan 同时包含 Proxy 和 CE 任务，会用独立 `ceStream` + `cudaEvent` 并行执行（`src/rma/rma.cc:24–88`）。

---

## 4. CE 路径实现

主要代码：`src/rma/rma_ce.cc`

### 4.1 初始化

`ncclRmaCeInit` 为每个 RMA context 分配一块对称信号缓冲区，布局如下（`src/include/rma/rma_ce.h:35–41`）：

```
[0 .. nRanks-1]              非 graph 每 rank signal
[nRanks]                     非 graph 聚合 signal
[nRanks+1 .. 2*nRanks]       graph 每 rank signal
[2*nRanks+1]                 graph 聚合 signal
[2*nRanks+2 .. 3*nRanks+1]   graph 每 rank ack flag
```

graph 区域的 ack flag 初始化为 1。同时创建独立的 `ceStream` 和 `ceEvent`。

### 4.2 Put 执行

`ncclRmaCePutLaunch` 分持久（graph）与非持久：

**非持久**（`ncclRmaCePutLaunchNonPersist`）：

- 按 peer 组织任务队列，每轮每个 peer 最多处理一个任务，避免同一 signal 地址在 CE batch 内出现多次。
- 数据拷贝通过 `ncclCeLaunchBatchOps`。
- signal 先用 `cuStreamBatchMemOp` 把递增序列号写到 `signalOpSeqsDev`，再用 CE 拷贝到对端 `signalsDev[rank]`。

**持久/graph**（`ncclRmaCePutLaunchPersist`）：

- ack handshake：`waitValue(ack==1)` + `writeValue(ack=0)`。
- 数据拷贝到对端窗口。
- signal 写 `graphSignalsDev[rank] = 1`。

### 4.3 Wait 执行

`ncclRmaCeWaitLaunch`：

- **非持久**：对每个 peer 用 `CU_STREAM_MEM_OP_WAIT_VALUE_64` 等待 `signalsDev[peer]` ≥ 期望值。
- **持久**：`waitValue(graphSignal==1)` → `writeValue(graphSignal=0)` → 向对端写 ack=1，完成 wait/reset/ack 握手。

---

## 5. Proxy 路径实现

主要代码：

- `src/rma/rma_proxy.cc`：生命周期、context、连接、progress 线程。
- `src/rma/rma_proxy_launch.cc`：descriptor 构建、enqueue、stream memop 参数。
- `src/rma/rma_proxy_progress.cc`：proxy 线程推进网络操作。

### 5.1 插件接口

RMA 插件 vtable 定义：`src/include/plugin/rma/rma_v14.h`

```c
init / devices / getProperties / listen / connect / createContext
regMrSym / regMrSymDmaBuf / deregMrSym
iput / iputSignal / iflush / test / rmaProgress / queryLastError
```

### 5.2 插件加载

`src/plugin/rma.cc` 在 `ncclRmaInit` 时按以下优先级加载：

1. 环境变量 `NCCL_RMA_PLUGIN` 指定的外部 so。
2. GIN 插件若提供 RMA 支持。
3. NET 插件若提供 RMA 支持。
4. 内部 IB RMA 插件 `ncclRmaIbProxy`。

### 5.3 连接与 Context

`ncclRmaProxyConnectOnce`（`src/rma/rma_proxy.cc:374`）：

- 通过 allGather 取所有 rank 本地 RMA 设备数的最小值。
- 创建物理 RMA collective communicator。
- 为每个 `numRmaCtx` 创建虚拟 `ncclRmaProxyCtx`，round-robin 映射到物理 comm。
- 启动独立的 RMA Proxy Progress Thread。

### 5.4 Proxy Context 数据结构

`ncclRmaProxyCtx`（`src/include/rma/rma_proxy.h:105`）关键字段：

- `signalsDev` / `signalsMhandle`：GPU 信号内存及 MR。
- `opSeqs` / `readySeqs` / `doneSeqs`：每 peer 序列号，CPU 可访问。
- `circularBuffers`：每 peer lock-free 环形 pending 队列。
- `inProgressQueues`：每 peer 在途 descriptor 队列。
- `persistentQueues` / `cpuAccessSignals` / `flushBufDev`：graph 模式专用。

### 5.5 Descriptor 协议

Proxy 使用 4 步 descriptor 协议（`src/include/rma/rma_proxy.h:220–229`）：

1. `BuildDesc`：分配并填充 `ncclRmaProxyDesc`。
2. `{Put,PutGroup,Wait}Params`：把字段 snapshot 到 `CUstreamBatchMemOpParams`。
3. `EnqueueDesc`：把 descriptor 所有权转给 proxy 队列。
4. `ncclCuStreamBatchMemOp`：在用户 stream 上发出 memop。

Descriptor 类型：

- `ncclRmaDescTypePutSignal`：单个 put + 可选 signal。
- `ncclRmaDescTypePutSignalGroup`：一组 put-signal。
- `ncclRmaDescTypeWaitSignal`：等待多 peer signal。

### 5.6 非持久 Put 的触发/完成

`ncclRmaProxyPutLaunch`（`src/rma/rma_proxy_launch.cc:551`）：

- 为每个 Proxy task 创建 descriptor。
- **Start op**：`writeValue(readySeqDev, opSeq)`，通知 proxy 可以开始。
- **Done op**：`waitValue(doneSeqDev, opSeq)`，GPU stream 阻塞直到 proxy 标记完成。

`ncclRmaProxyProgress`（`src/rma/rma_proxy_progress.cc:269`）：

- 检查 `readySeq >= opSeq`。
- 调用 `iput` / `iputSignal` 发出网络请求，挂到 `inProgressQueues`。
- 请求完成后，用 release 语义写 `doneSeq = opSeq`。

### 5.7 Graph / 持久模式

Graph 下 descriptor 挂在 `persistentQueues` 上，直到 graph 被 reclaim：

- Start：`readySeq = 1` 触发网络操作，然后 `readySeq = 0`，状态切到 `InProgress`。
- Done：网络完成后 `doneSeq = 1`，GPU 等待后可选重置为 0，状态回到 `Ready` 供下一次 replay。
- WaitSignal：proxy 轮询 `cpuAccessSignals[peer]`，到达阈值后必要时 `iflush` 刷 NIC-GPU 路径，再写 `doneSeq`。

Graph 销毁时 `src/enqueue.cc:1512` 调用 `ncclRmaProxyReclaimPlan`，会暂停 proxy 线程、删除属于该 plan 的 persistent descriptor、再恢复线程。

---

## 6. 信号与排序

- Signal 模式：当前仅 `NCCL_SIGNAL`（累加 1）或 `NCCL_SIGNAL_NONE`。
- Proxy 路径的 signal 通过插件 `iputSignal` 写入，offset 为 `comm->rank * sizeof(uint64_t)`。
- CE 路径的 signal 写到对称窗口信号区。
- 非 graph 模式下，CE 用递增序列号避免同一 batch 内对同一 signal 地址的并发写。
- Graph 模式下，CE 通过 ack 握手保证可重复捕获。

---

## 7. 内存注册

RMA 数据缓冲区来自用户通过 `ncclCommWindowRegister` 注册的对称窗口。

Proxy 路径在连接时通过 `ncclRmaProxyRegMrSym` 把 GPU 缓冲区注册到 RMA 插件（`src/rma/rma_proxy.cc:63`）：

- 优先尝试 DMA-BUF（`getDmaBufFd`）。
- 失败则回退到普通 `regMrSym`。
- HOST 指针直接注册；CUDA 指针先尝试 DMA-BUF。

---

## 8. 关键文件速查

| 文件 | 作用 |
|------|------|
| `src/nccl.h.in` | 公开 API |
| `src/collectives.cc` | API 入口，构造 `ncclInfo` |
| `src/enqueue.cc` | 创建 RMA task、生成 plan |
| `src/group.cc` | group end 调度并调用 `ncclLaunchRma` |
| `src/rma/rma.cc` | 顶层调度：CE/Proxy 并行、任务拆分 |
| `src/rma/rma_ce.cc` | CE 路径实现 |
| `src/rma/rma_proxy.cc` | Proxy 生命周期、context、连接、progress 线程 |
| `src/rma/rma_proxy_launch.cc` | descriptor 构建、enqueue、stream memop |
| `src/rma/rma_proxy_progress.cc` | proxy 线程推进网络操作 |
| `src/plugin/rma.cc` | RMA 插件加载、init/finalize |
| `src/include/plugin/rma/rma_v14.h` | RMA 插件 vtable |
| `src/include/rma/rma_proxy.h` | Proxy descriptor/context 数据结构 |
| `src/include/rma/rma_ce.h` | CE 路径数据结构 |

---

## 9. 一句话总结

NCCL RMA = **基于对称窗口的单边 put/signal/wait**：同节点走 GPU CE 直传 + 显存信号，跨节点走 RMA 网络插件 + CPU Proxy 线程，通过 `readySeq/doneSeq` 和 `cuStreamBatchMemOp` 实现 GPU stream 与 Proxy 的异步协同，并完整支持 CUDA Graph 的持久描述符 replay。
