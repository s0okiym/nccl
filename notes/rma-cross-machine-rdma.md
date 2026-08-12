# NCCL RMA 跨机通信的 RDMA 网络封装

## 1. 概述

NCCL 的跨机 RMA 通过 **RMA 插件层 + 内部 IB RMA Proxy** 实现。核心思路是：

- 在 RMA 插件 vtable 中定义单边 `put/signal/get/flush` 语义。
- 内部 `ncclRmaIbProxy` 复用 NCCL 的 IB verbs 传输基础设施。
- 通过 full-mesh QP、MR 注册、allGather 交换 base VA + rkey，把 RMA 语义映射到 `RDMA_WRITE / ATOMIC_FETCH_ADD / RDMA_READ`。
- 由 Proxy 线程负责 post send 和 poll CQ，完成与 GPU stream 的异步同步。

---

## 2. RMA 插件抽象层

RMA 插件 vtable：`src/include/plugin/rma/rma_v14.h`

```c
init / devices / getProperties / listen / connect / createContext
regMrSym / regMrSymDmaBuf / deregMrSym
iput / iputSignal / iget / iflush
test / rmaProgress / queryLastError / finalize
```

NCCL Proxy 路径（`src/rma/rma_proxy_progress.cc`）只调用这些回调，不感知底层网络。

插件加载顺序（`src/plugin/rma.cc:157–218`）：

1. 外部 so（`NCCL_RMA_PLUGIN`）
2. GIN 插件若提供 RMA
3. NET 插件若提供 RMA
4. 内部 `ncclRmaIbProxy`（`src/transport/net_ib/gin.cc:810`）——兜底实现

---

## 3. 内部 IB RMA Proxy

`ncclRmaIbProxy` 定义：`src/transport/net_ib/gin.cc:810`

```c
ncclRma_t ncclRmaIbProxy = {"RMA_IB_PROXY",
    ncclRmaIbProxyInit,
    ncclIbDevices,
    ncclRmaIbProxyGetProperties,
    ncclIbListen,
    ncclRmaIbProxyConnect,
    ncclRmaIbProxyCreateContext,
    ncclRmaIbProxyRegMrSym,
    ncclRmaIbProxyRegMrSymDmaBuf,
    ncclRmaIbProxyDeregMrSym,
    ncclRmaIbProxyDestroyContext,
    ncclGinIbCloseColl,
    ncclIbCloseListen,
    ncclRmaIbProxyIPut,
    ncclRmaIbProxyIPutSignal,
    ncclRmaIbProxyIGet,
    ncclRmaIbProxyIFlush,
    ncclRmaIbProxyTest,
    NULL, NULL,
    ncclGinIbFinalize};
```

它本质上是在 NCCL IB net 传输之上再包一层 RMA 语义。

---

## 4. 跨机连接：建立全互联 QP

### 4.1 初始连接

`ncclRmaIbProxyConnect`（`gin.cc:348`）复用 `ncclGinIbConnect`（`gin.cc:174`），先建立一对 send/recv comm：

- `ncclIbConnectImpl`：本 rank 连向下一个 rank。
- `ncclIbAcceptImpl`：本 rank 接受上一个 rank 的连接。

这一步主要是为了借用 GIN/IB 的 allGather 能力来交换后续 QP handle。

### 4.2 RMA Context：全互联 QP Mesh

真正的跨机 RMA 需要每个 rank 与每个 peer rank 都有直接 QP。`ncclRmaIbProxyCreateContext`（`gin.cc:366`）为每个 context 创建全互联：

```c
for (int i = 0; i < nranks; i += config->rankStride) {
    int connectPeer = (cComm->rank + i) % nranks;
    int acceptPeer  = (cComm->rank - i + nranks) % nranks;
    // 建立到 connectPeer 的 send QP
    ncclIbConnectImpl(..., &gc->fullSendComm[connectPeer], ...);
    // 接受来自 acceptPeer 的 recv QP
    ncclIbAcceptImpl(lComm, &gc->fullRecvComm[acceptPeer], ...);
    ncclGinIbP2PBarrier(cComm);  // 同步，防止连接风暴
}
```

结果：

- `rmaProxyCtx->fullSendComm[peer]`：到 peer 的发送 QP。
- `rmaProxyCtx->fullRecvComm[peer]`：来自 peer 的接收 QP。

这是跨机 RMA 的物理基础——任意两 rank 之间可以直接 post RDMA 操作，无需中间节点转发。

---

## 5. 内存注册与地址交换

### 5.1 注册本地 MR

`ncclRmaIbProxyRegMrSymDmaBuf`（`gin.cc:450`）：

```c
ncclIbRegMrDmaBufInternal(cComm->recvComm, data, size, type, offset, fd, mr_flags,
                          (void**)&rmaMrHandle->mrHandle);
```

内部调用 `ibv_reg_mr`，访问权限包含：

```c
IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC
```

允许对端通过 RDMA_WRITE / RDMA_READ / ATOMIC 访问这块内存。

### 5.2 交换 base VA 和 rkey

注册后，每个 rank 把自己的 `data` 指针和 `mr->rkey` allGather 到所有 rank：

```c
cComm->allGather(cComm, &data, rmaMrHandle->base_vas, sizeof(uintptr_t));
cComm->allGather(cComm, &mrHandle->mrs[0]->rkey, rmaMrHandle->rkeys, sizeof(uint32_t));
```

`ncclRmaIbProxyMrHandle` 结构（`gin.cc:332`）：

```c
struct ncclRmaIbProxyMrHandle {
  struct ncclIbMrHandle* mrHandle;  // 本地 lkey + MR
  uintptr_t* base_vas;              // 每个 rank 的 base VA
  uint32_t* rkeys;                  // 每个 rank 的 rkey
};
```

发 put 时：本地用 `base_vas[myRank] + offset` + 本地 `lkey`；对端用 `base_vas[peer] + offset` + `rkeys[peer]`。

---

## 6. RMA 操作到 IB Verbs 的映射

### 6.1 Put → RDMA_WRITE

`ncclRmaIbProxyIPut`（`gin.cc:521`）：

```c
wr.opcode      = IBV_WR_RDMA_WRITE;
wr.send_flags  = IBV_SEND_SIGNALED;
wr.wr.rdma.remote_addr = (uint64_t)dstPtr;  // 对端 VA
wr.wr.rdma.rkey        = rkey;              // 对端 rkey
sge.addr = (uintptr_t)srcPtr;               // 本地 VA
sge.lkey = lkey;                            // 本地 lkey
sge.length = size;
wrap_ibv_post_send(qp->qp, &wr, &bad_wr);
```

标准的 RDMA Write，源数据在本地 GPU 内存，直接写到对端 GPU 内存，两端 CPU 不参与数据搬运。

### 6.2 Put + Signal → RDMA_WRITE + ATOMIC_FETCH_ADD

`ncclRmaIbProxyIPutSignal`（`gin.cc:625`）把 data 和 signal 串成两个 WR：

```c
// WR[0]: 数据 PUT
wr[0].opcode = IBV_WR_RDMA_WRITE;
wr[0].send_flags = 0;              // 不需要 CQE，靠 signal 的 CQE 感知完成
wr[0].next = &wr[1];

// WR[1]: signal 原子累加
wr[1].opcode = IBV_WR_ATOMIC_FETCH_AND_ADD;
wr[1].send_flags = IBV_SEND_SIGNALED;
wr[1].wr.atomic.remote_addr = signalPtr;       // 对端 signal 地址
wr[1].wr.atomic.compare_add = signalValue;     // 通常为 1
wr[1].wr.atomic.rkey        = signalRkey;
sge[1].addr = (uintptr_t)&comm->putSignalScratchpad;  // 本地 scratchpad
```

关键点：

- data WR 不发 CQE，signal WR 发 CQE；IB 硬件保证同一 QP 上 WR 按序执行，因此 signal 完成意味着 data 也已完成。
- signal 使用 `IBV_WR_ATOMIC_FETCH_AND_ADD`，实现对端内存原子 `+1` 或 `+signalValue`。
- 本地 SGE 是 `putSignalScratchpad`，因为 fetch-add 会读旧值回来，但这里只关心对端内存被原子修改。

### 6.3 Get → RDMA_READ

`ncclRmaIbProxyIGet`（`gin.cc:573`）：

```c
wr.opcode = IBV_WR_RDMA_READ;
wr.send_flags = IBV_SEND_SIGNALED;
wr.wr.rdma.remote_addr = remotePtr;
wr.wr.rdma.rkey        = rkey;
sge.addr = (uintptr_t)localPtr;
sge.lkey = lkey;
```

把对端 GPU 内存读回本地。

### 6.4 Flush → RDMA_READ

`ncclRmaIbProxyIFlush`（`gin.cc:768`）向已注册内存发一次 RDMA_READ：

```c
wr.opcode = IBV_WR_RDMA_READ;
wr.send_flags = IBV_SEND_SIGNALED;
```

目的不是搬数据，而是用一次带 CQE 的 RDMA 操作把 NIC-GPU 路径刷干净，确保之前通过 NIC 写入的数据对本地 GPU 可见。对应 proxy progress 中 graph 模式下 waitSignal 成功后的 `ncclRmaProxyFlushNicGpuPath`（`rma_proxy_progress.cc:154`）。

---

## 7. 完成通知：CQ Polling

所有 RMA 操作都是异步的，完成通过 `ncclRmaIbProxyTest`（`gin.cc:709`）轮询 CQ：

```c
wrap_ibv_poll_cq(devBase->cq, 4, wc, &wrDone);
for (int i = 0; i < wrDone; i++) {
    struct ncclIbRequest* wcReq = commBase->reqs + wc[i].wr_id;
    wcReq->events[0]--;
    if (wcReq == req && wcReq->events[0] == 0) {
        *done = 1;
        ncclIbFreeRequest(wcReq);
    }
}
```

- post send 时 `ncclIbAddEvent` 增加 `req->events[devIndex]`。
- 收到 CQE 后递减；减到 0 表示该请求的所有 WR 都已完成。
- `ncclRmaProxyProgress`（`rma_proxy_progress.cc`）不断调用 `ncclRma->test` 驱动状态机。

---

## 8. 跨机 RMA 为什么能工作

1. **连接**：每个 rank 与每个对端 rank 建立直接 QP（full mesh），跨机流量直接走 IB 网络，不经过 CPU 拷贝。
2. **内存**：用户窗口通过 `ncclCommWindowRegister` 注册成 symmetric window；NCCL 用 `ibv_reg_mr` 把它 pin 住并允许 `REMOTE_WRITE/READ/ATOMIC`。
3. **地址交换**：通过 allGather 把每个 rank 的 base VA 和 rkey 广播到所有 rank，任何 rank 都能构造对端地址。
4. **单边操作**：
   - `put` → `RDMA_WRITE`
   - `signal` → `ATOMIC_FETCH_ADD`
   - `get` / `flush` → `RDMA_READ`
5. **顺序保证**：同一 QP 上 WR 按序执行；`put+signal` 把 data WR 设为无 CQE、signal WR 设为 signaled，靠 signal 的完成推断 data 完成。
6. **CPU 协调**：Proxy 线程负责 post send、poll CQ、推进 descriptor 状态机，通过 `readySeq/doneSeq` 与 GPU stream 同步。

---

## 9. 关键源码速查

| 文件/函数 | 作用 |
|-----------|------|
| `src/include/plugin/rma/rma_v14.h` | RMA 插件 vtable |
| `src/plugin/rma.cc` | 加载 RMA 插件 |
| `src/transport/net_ib/gin.cc:810` | `ncclRmaIbProxy` 定义 |
| `gin.cc:366` `ncclRmaIbProxyCreateContext` | 创建全互联 QP mesh |
| `gin.cc:450` `ncclRmaIbProxyRegMrSymDmaBuf` | 注册 MR 并 allGather base VA / rkey |
| `gin.cc:521` `ncclRmaIbProxyIPut` | `RDMA_WRITE` |
| `gin.cc:625` `ncclRmaIbProxyIPutSignal` | `RDMA_WRITE` + `ATOMIC_FETCH_ADD` |
| `gin.cc:573` `ncclRmaIbProxyIGet` | `RDMA_READ` |
| `gin.cc:768` `ncclRmaIbProxyIFlush` | 用于 flush 的 `RDMA_READ` |
| `gin.cc:709` `ncclRmaIbProxyTest` | CQ 轮询完成 |
| `src/transport/net_ib/reg.cc:71` `ncclIbRegMrDmaBufInternal` | 底层 `ibv_reg_mr` 封装 |
| `src/transport/net_ib/p2p.cc:43` `ncclIbAddEvent` | 请求事件计数 |

---

## 10. 一句话总结

NCCL 跨机 RMA 的封装方式是：**在 RMA 插件层定义单边 put/signal/get/flush 语义，在内部 `ncclRmaIbProxy` 中通过 full-mesh QP、MR 注册 + 全交换 base VA/rkey，把语义映射到 `RDMA_WRITE / ATOMIC_FETCH_ADD / RDMA_READ`，最终由 Proxy 线程 post 并 poll CQ，从而用标准 IB RDMA 实现跨机器的 GPU 内存单边访问。**
