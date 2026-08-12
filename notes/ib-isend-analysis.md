# ncclIbIsend 调用链与耗时点分析

本文整理 `src/transport/net_ib/p2p.cc` 中 `ncclIbIsend` 的调用链、发送内容、接收方未准备好时的轮询行为，以及内部可能的耗时点。

## 1. 函数定位

`ncclIbIsend` 是 NCCL IB/RoCE 网络插件的异步发送入口，定义于 `src/transport/net_ib/p2p.cc:262`，通过 `ncclNet_t ncclNetIb` 注册到 NCCL core：

```c
// src/transport/net_ib/common.cc:193-216
ncclNet_t ncclNetIb = {
  "IB",
  ...
  ncclIbIsend,   // line 204
  ...
};
```

## 2. 调用链

### 2.1 主调用路径（正常 collective 数据发送）

```text
ncclProxyProgress()              [src/proxy.cc:954]
  └── progressOps()              [src/proxy.cc:980]
        └── sendProxyProgress()  [src/transport/net.cc:1304]
              └── proxyState->ncclNet->isend(...)
                    └── (src/plugin/net/net_v*.cc 适配)
                          └── ncclIbIsend()          [src/transport/net_ib/p2p.cc:262]
                                └── ncclIbMultiSend() [src/transport/net_ib/p2p.cc:83]
                                      └── wrap_ibv_post_send()
```

`sendProxyProgress` 在 net transport 初始化时被挂到 `send->proxyConn.proxyProgress`：`src/transport/net.cc:528-534`。

### 2.2 其他调用方

- **Bootstrap**：`src/bootstrap.cc:171` 的 `netIsend()` 在初始化阶段发送 OOB 控制消息。
- **GIN**：`src/transport/net_ib/gin.cc:122` 的 `ncclGinIbAllGather()` 用于 GPU-Initiated Network。

## 3. 发送内容

`ncclIbIsend` 的 payload 是 `data` 指向的 `size` 字节数据，通常是某个 channel 某个 step 的 send buffer slice：

- LL/LL128：指向 LL FIFO 中的一段。
- SIMPLE：指向 `localBuff + buffSlot * stepSize` 或 registered user buffer。

实际网络动作在 `ncclIbMultiSend()` 中完成：

- 构造 `IBV_WR_RDMA_WRITE` WR，把数据写到对端 `ctsFifo[slot][r].addr`。
- 最后一个 WR 使用 `IBV_WR_RDMA_WRITE_WITH_IMM`，`imm_data` 携带 request ID（BY_ID 匹配）或发送 size（BY_INDEX 匹配）。
- 多 recv（`nreqs > 1`）时，还会把各 request 的 size 写到对端 `remCmplsRecords`。

## 4. 接收方未 ready 时的轮询链

### 4.1 发送方提前返回

`ncclIbIsend` 首先检查 CTS FIFO：

```c
// src/transport/net_ib/p2p.cc:282-285
if (slots[0].idx != idx) {
    *request = NULL;
    return ncclSuccess;
}
```

如果对端还没 post recv，`slots[0].idx` 不等于期望值 `idx`，函数立即返回，`*request = NULL`。

### 4.2 上层重试

回到 `sendProxyProgress`：`src/transport/net.cc:1414` 附近。由于 `sub->requests[buffSlot] == NULL`，`sub->transmitted` 不会推进，这个 step 下次 proxy tick 继续重试。

### 4.3 Proxy 线程主循环

```text
ncclProxyProgress()
  ├── progressOps()  → 调用所有 active op 的 progress（包括 sendProxyProgress）
  ├── 若全部 idle，std::this_thread::yield()
  └── ncclProxyGetPostedOps() 拉新 op
```

因此，**发送方 proxy 线程会不断轮询 `ncclIbIsend`，直到对端把 CTS 写过来**。

### 4.4 接收方何时 ready

接收方 proxy 线程：

```text
ncclProxyProgress()
  └── recvProxyProgress()     [src/transport/net.cc:1470]
        └── ncclNet->irecv()
              └── ncclIbIrecv() [src/transport/net_ib/p2p.cc:430]
                    └── ncclIbPostFifo() [src/transport/net_ib/p2p.cc:364]
                          └── wrap_ibv_post_send(IBV_WR_RDMA_WRITE)
```

`ncclIbIrecv` 把 `(addr, rkeys, tag, size, idx)` 写入本地 `remCtsFifo.elems[slot]`，然后通过一次 RDMA WRITE 写到发送方的 `ctsFifo[slot]`。发送方下一次轮询看到 `slots[0].idx == idx` 后才能真正发数据。

## 5. 内部耗时点分析

### 5.1 CTS metadata 等待循环

```c
for (int r = 1; r < nreqs; r++) {
    while (slots[r].idx != idx);
}
std::atomic_thread_fence(std::memory_order_seq_cst);
```

- 这是一个**忙等 + 全内存屏障**。
- 正常路径下几乎立即退出，因为对端 `ncclIbPostFifo` 是一次性把整个 `nreqs` 个 entry RDMA 写过来的。
- 如果这里耗时高，通常意味着对端 CTS 写延迟，或者 CPU cache coherency 延迟。它本身是症状，不是根因。

### 5.2 `wrap_ibv_post_send()`

`ncclIbMultiSend` 中真正下发 IB WR 的地方：

```c
NCCLCHECK(wrap_ibv_post_send(qp->qp, comm->wrs, &bad_wr));
```

这是 `ncclIbIsend` 内部**唯一可能真实耗时较长**的地方，原因包括：

- QP send queue 接近满。
- 高频小消息 burst（如每 step 插 100 次 allreduce）。
- 多 QP / multi-rail 场景下循环 post 多次。
- HCA firmware 处理延迟。

### 5.3 其他操作

- `ncclIbGetRequest()`：扫描请求池，通常很快。
- `ncclIbCommBaseGetQpForRequest()` / `ncclIbCommBaseGetNqpsPerRequest()`：纯数组索引计算。
- lkey 拷贝、本地 `remCmplsRecords` 更新：通常很快。

### 5.4 真正的端到端延迟通常不在 `ncclIbIsend`

`ncclIbIsend` 是异步的，只负责把 WR 提交给 HCA。真正的发送延迟往往发生在：

- `ncclIbTest()` 等待 CQ completion（对端没收完 / 网络慢）。
- 接收方 `ncclIbIrecv` 延迟 post recv（GPU 没 ready / proxy 线程调度慢）。
- 网络/fabric 传输延迟。

## 6. 计时辅助代码

若要在 `ncclIbIsend` 内部分段计时，可用 `clock_gettime`：

```c
#include <time.h>

static inline uint64_t getMicroSeconds() {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000ULL + ts.tv_nsec / 1000;
}
```

分段插桩示例：

```c
uint64_t t0 = getMicroSeconds();

// 段 1：CTS 等待
if (slots[0].idx != idx) { *request = NULL; return ncclSuccess; }
for (int r = 1; r < nreqs; r++) while (slots[r].idx != idx);

uint64_t t1 = getMicroSeconds();

// 段 2：构造 request
// ... ncclIbGetRequest / lkeys ...

uint64_t t2 = getMicroSeconds();

// 段 3：ncclIbMultiSend / ibv_post_send
NCCLCHECK(ncclIbMultiSend(comm, slot));

uint64_t t3 = getMicroSeconds();

if (t1 - t0 > 1000 || t3 - t2 > 1000) {
    INFO(NCCL_NET, "ncclIbIsend breakdown(us): cts_wait=%lu build=%lu post_send=%lu",
         t1 - t0, t2 - t1, t3 - t2);
}
```

> 注：NCCL 源码中不存在 `ncclCpuMicroSecond` 函数，之前讨论中使用的 `getMicroSeconds()` 是自定义辅助函数。
