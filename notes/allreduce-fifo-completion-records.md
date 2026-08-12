# NCCL AllReduce 关键 FIFO 结构（host↔device↔proxy）与 IB completion records 注册分析

> 分析对象：`src/include/comm.h`、`src/include/device.h`、`src/include/proxy.h`、`src/include/collectives.h`、
> `src/init.cc`、`src/enqueue.cc`、`src/proxy.cc`、`src/device/common.h`、`src/device/prims_simple.h`、
> `src/transport/p2p.cc`、`src/transport/net.cc`、`src/transport/net_ib/{p2p.cc,connect.cc,common.h}`

本文回答两个问题：

1. `src/transport/net_ib/p2p.cc` `ncclIbMultiSend` 里的 `comm->remCmplsRecords.elems` 是不是注册（register MR）过的内存？
2. AllReduce 实现中有哪些关键 FIFO 结构？host 与 device、device 与 proxy 之间如何交互？

---

## 第一部分：`ncclIbMultiSend` 中 `comm->remCmplsRecords.elems` 是否注册过？

**结论：是。它是发送端本地的一块 host 内存，在连接建立阶段对每个本地 IB 设备 PD 调用了 `ibv_reg_mr` 注册；
但它只是"本地 shadow"，RDMA 写入的真正目标是接收端的 `cmplsRecords`（同样是注册内存），
`addr/rkeys` 由连接握手时经 socket metadata 交换得到。即这条 RDMA 写的源和目标两端都是注册内存。**

### 1.1 定义与分配

`struct ncclIbRemCompletionsRecords`（`src/transport/net_ib/common.h:309-321`）：

```cpp
struct ncclIbRemCompletionsRecords {
  // 接收端 completion records 在发送端的本地 shadow，发送端把要写/读到对端
  // completion records 的内容先放在这里。
  int elems[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS];
  uint64_t addr;                              // 接收端 completion records 的远端地址
  uint32_t rkeys[NCCL_IB_MAX_DEVS_PER_NIC];   // 接收端每个设备的 rkey
};
```

它内嵌在 `struct ncclIbSendComm`（common.h:451）里，发送 comm 在 `connect.cc:823` 用
`ncclIbMalloc((void**)&comm, sizeof(struct ncclIbSendComm))` 一次性分配，是普通 host 内存。

### 1.2 注册动作（本地源 buffer）

`ncclIbConnectImpl` 在交换完对端 metadata 之后、QP 转 RTS 之前（`connect.cc:1064-1071`）：

```cpp
for (int i = 0; i < comm->base.vProps.ndevs; i++) {
  ncclIbSendCommDev* commDev = comm->devs + i;
  NCCLCHECKGOTO(wrap_ibv_reg_mr(&commDev->cmplsRecordsMr, comm->devs[i].base.pd,
                                &comm->remCmplsRecords.elems, sizeof(comm->remCmplsRecords.elems),
                                IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ),
                ret, fail);
  comm->devs[i].sge.lkey = comm->devs[i].cmplsRecordsMr->lkey;  // lkey 预存到每设备的 sge
}
```

对每个本地 merged device 的 PD 各注册一次，lkey 预存到 `comm->devs[i].sge.lkey`；
对应 dereg 在 `connect.cc:1674`（dtor 路径）。

### 1.3 在 `ncclIbMultiSend` 里的用法（`src/transport/net_ib/p2p.cc`）

- `ncclIbIsend` 把各 sub-request 的 size 写进本地 shadow：
  `comm->remCmplsRecords.elems[slot][r] = req->send.size;`（p2p.cc:339）
- 多 recv（`nreqs > 1`）时，最后一个 WR 是 `IBV_WR_RDMA_WRITE_WITH_IMM`（p2p.cc:137-145, 190-198）：
  - 本地 gather 源：`lastWr->sg_list = &(comm->devs[devIndex].sge);
    lastWr->sg_list[0].addr = (uint64_t)(comm->remCmplsRecords.elems[slot]);`（p2p.cc:194-195）
    —— sge.lkey 用的就是 1.2 中注册的 `cmplsRecordsMr->lkey`，注释写明
    "the lkey is already correct from the initialization phase"（p2p.cc:193）。
  - 远端目标：`lastWr->wr.rdma.remote_addr = comm->remCmplsRecords.addr
    + slot * sizeof(struct ncclIbRequestCompletionRecord);`（p2p.cc:139）
  - 远端 rkey：`lastWr->wr.rdma.rkey = comm->remCmplsRecords.rkeys[devIndex];`（p2p.cc:198）

### 1.4 `addr/rkeys` 指向的对端内存也是注册的

接收端 `ncclIbRecvComm::cmplsRecords[NET_IB_MAX_REQUESTS]`（common.h:521，真实的
`struct ncclIbRequestCompletionRecord` 数组，含 `sizes[]` 与 `completions[]`，common.h:176-189）：

- 接收端在 accept 阶段注册（`connect.cc:1581-1585`，权限
  `IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ`）；
- 然后随 metadata 发回：`meta.devs[i].rkey = rCommDev->cmplsRecordsMr->rkey;`（connect.cc:1585）、
  `meta.addr = (uint64_t)rComm->cmplsRecords;`（connect.cc:1611）；
- 发送端存下对端信息：`comm->remCmplsRecords.addr = remMeta.addr;`、
  `comm->remCmplsRecords.rkeys[i] = remMeta.devs[i].rkey;`（connect.cc:1054-1056）。

接收端 proxy 之后读这些 `sizes` 来上报 recv 完成（p2p.cc:637、765-766）。

### 1.5 小结

| 角色 | 内存 | 注册点 | 用途 |
|------|------|--------|------|
| 发送端本地 | `remCmplsRecords.elems`（shadow，host 内存） | connect.cc:1066 `wrap_ibv_reg_mr`，lkey → `devs[i].sge.lkey` | RDMA 写的本地 gather 源（各 sub-req 的 size） |
| 接收端远端 | `rComm->cmplsRecords`（真实数组） | connect.cc:1581 `wrap_ibv_reg_mr`，rkey/addr 经 metadata 交换 | RDMA 写的目标，接收端读 size 上报完成 |

---

## 第二部分：AllReduce 关键 FIFO 结构与 host/device/proxy 信息交互

NCCL 一次 allreduce（group launch）涉及三层 FIFO，层层解耦：

```
host 主线程                    device kernel                 proxy 服务线程
    │ ① workFifoBuf 写入 work       │                              │
    ├──────────────────────────────▶│                              │
    │ ② proxy ops pool (共享内存)    │                              │
    ├───────────────────────────────┼─────────────────────────────▶│
    │                               │ ③ sendMem/recvMem + connFifo │
    │                               │◀──── head/tail/size 信用 ───▶│（或 P2P 直写对端 GPU）
    │ ④ event 回调推进 consumed      │  workCounter 写回             │
    │◀──────────────────────────────┤                              │
```

### 2.1 Host → Device：work FIFO（`comm->workFifoBuf`）

**结构与分配**（`src/include/comm.h:662-670`、`src/init.cc:632-642`）：

- `workFifoBuf` / `workFifoBufDev` / `workFifoBytes`（2 的幂字节环形缓冲）；
- 优先分配 GDR 映射的 CUDA 内存（`ncclGdrCudaCalloc`，kernel 直接读显存里的 work），
  否则 pinned host 内存（`ncclCudaHostCalloc`，UVA 直读）；
- 游标：`workFifoProduced`（host 写的单调字节位置）、`workFifoConsumed`（kernel 已消费位置）。

**内容**：batch 头 `ncclDevWorkBatch`（`src/include/device.h:392-410`）+ 各 work 结构
（`ncclDevWorkColl` / `ncclDevWorkP2p` / …）。batch 头字段：

- `nextJump / nextExtends`：batch 链表跳转与合并；
- `workType / funcId`：work 类型与 kernel 函数 id；
- `offsetBase + offsetBitset`：本 channel 的 work 在 FIFO 中的位置（基址 + 位图）。

**写入**（`uploadWork()`，`src/enqueue.cc:1248-1318`）：

- 三种存储模式：随 kernel 参数（≤4K，`ncclDevWorkStorageTypeArgs`）、
  FIFO（默认，`ncclDevWorkStorageTypeFifo`）、Persistent（graph capture 独立 buffer）；
- FIFO 模式下按 16B 拷贝、更新 `comm->workFifoProduced`，GDR 情形加 `wc_store_fence()`；
- 背压：`waitWorkFifoAvailable`（enqueue.cc:1216-1232），
  `produced - consumed > fifoBytes` 时自旋并驱动事件回调。

**Device 读取**（`src/device/common.h`）：

- kernel 参数 `ncclDevKernelArgs{comm, channelMask, workStorageType, workMask, workBuf}`
  （device.h:473-481）；
- 每个 block 按 `channelMask` 中第 n 个置位认领 channel（blockIdx.x ↔ channelId 映射）；
- 以 16B（ulonglong2）为单位把 batch/work 搬进 shared memory（common.h:228-238），
  沿 `nextJump` 遍历 batch 链表（common.h:242-256），执行 `RunWorkColl` 等。

**Device 回写与 host 回收**：

- 每个 channel 维护 `workCounter`：kernel 启动时从
  `ncclKernelCommAndChannels.channels[id].workCounter` 恢复（common.h:378-379），
  收尾时累加 `nWorks` 并写回（common.h:347-350）；profiler 另有
  `workStarted / workCompleted` 时间戳数组（common.h:329-353）；
- host **不轮询** device 计数器，而是用 CUDA event 回调：`ncclLaunchFinish`
  （enqueue.cc:1876-1898）在 FIFO 前进超过 `fifoBytes/8` 时 record event 并挂
  `KernelFinishCallback`，event 触发后 `comm->workFifoConsumed = 记录值`
  （enqueue.cc:1863-1873），由 `ncclCommPollEventCallbacks` 驱动；
- 注：`device.h:426` 的 `workFifoDone` 字段当前**只有声明、全仓库无使用点**，是遗留字段，
  实际机制即上述 workCounter + event 回调。

### 2.2 Host → Proxy：proxy ops pool（节点级共享内存 FIFO）

**结构**（`src/include/proxy.h:229-245`）：

```cpp
struct ncclProxyOpsPool {
  struct ncclProxyOp ops[MAX_OPS_PER_PEER * NCCL_MAX_LOCAL_RANKS];  // 定长槽位
  volatile int nextOps;      // 已投递链头（-1 为空）
  volatile int nextOpsEnd;   // 已投递链尾
  volatile int freeOps[NCCL_MAX_LOCAL_RANKS];  // 服务线程归还的空闲槽
  std::mutex mutex;
  std::condition_variable cond;
};
```

放在 ncclShm 共享内存，同节点所有 rank 的主线程与 proxy 服务线程共享。
`ncclProxyOp`（proxy.h:73-131）描述一条网络工作：connection、nbytes、opCount、pattern、
protocol、algorithm、send/recv buff 与 mhandle 等。

**生产**：enqueue 阶段对需要网络的 op 调 `ncclProxySaveOp`（`src/proxy.cc:602`；
调用点 `src/enqueue.cc:110`、`1422`）：

- 按 pattern（ring/tree/nvls/collnet…）展开为每 peer 一条 send/recv op
  （`SaveProxy`，proxy.cc:578-598）；
- `ncclLocalOpAppend`（proxy.cc:489-555）：从 free list 取槽（空则原子等 `freeOps`）、
  memcpy op、挂到本 rank 的 `proxyOps->nextOps..nextOpsEnd` 链；
  攒满 `MAX_OPS_PER_PEER` 时按 opCount 边界先截断投递一批。

**投递**：group 提交时 `ncclProxyStart`（proxy.cc:1014-1029）把各 peer 的链通过
`ncclProxyPost`（proxy.cc:477-487）交给 pool（mutex + cond 唤醒服务线程），并 `comm->opCount++`。

**消费**：proxy 服务线程 `ncclProxyProgress`（proxy.cc:954-1012）：

- 主循环 `progressOps` 推进 `state->active` 链表；
- idle 或达到 `NCCL_PROGRESS_APPEND_OP_FREQ` 频率时 `ncclProxyGetPostedOps`（proxy.cc:835-870）
  取回新投递的链（池空时在 cond 上等）；
- 逐条 `ProxyAppend`（proxy.cc:438-474）→ `ncclProxyOpToArgs` 转成 `ncclProxyArgs`
  （含 nsubs 个 `ncclProxySubArgs`，per-connection 状态机：
  `posted / transmitted / received / done` + `requests[NCCL_STEPS]`，proxy.h:140-183）；
  同 `opCount` 的 op 聚合成一个 args 的多个 subs 并行推进；
- args 完成后槽位经 `freeOps` 归还 pool。

### 2.3 Device ↔ 对端（GPU 或 proxy）：连接级 FIFO（`ncclSendMem`/`ncclRecvMem` + `ncclConnInfo`）

最核心的一层，kernel 的 LL/LL128/SIMPLE 协议全靠它做信用同步。

**结构**（`src/include/comm.h:53-77`、`src/include/collectives.h:72-77`）：

```cpp
struct ncclSendMem { uint64_t head; void* ptrExchange; uint64_t redOpArgExchange[2]; int offsFifo[NCCL_STEPS]; };
struct ncclRecvMem { uint64_t tail; struct ncclConnFifo connFifo[NCCL_STEPS]; int flush; };
struct ncclConnFifo { int mode; int offset; ssize_t size; void* ptr; };
```

协议数据缓冲按 stepSize 切 `NCCL_STEPS` 片，`head/tail` 是槽位信用计数器。

**抽象**：每个 (channel, peer, send/recv, connIndex) 一条连接；device 侧只见
`ncclConnInfo`（`src/include/device.h:133-151`）：

```cpp
struct ncclConnInfo {
  char* buffs[NCCL_NUM_PROTOCOLS];  // 协议数据缓冲（recv 本地 / send 远端）
  uint64_t* tail;                   // recv 本地 / send 远端
  uint64_t* head;                   // send 本地 / recv 远端
  struct ncclConnFifo* connFifo;    // "Used for GPU - Proxy communication"
  uint64_t step;                    // 当前步计数（跨 op 持久）
  ...
};
```

**P2P/NVLink 直连**（`src/transport/p2p.cc:568-569, 599-600`）：

- sendMem 在本 GPU，recvMem 在对端 GPU（IPC 映射）；
- 发送 kernel 直写对端 buff 后 `st_relaxed_sys_global(peer->tailPtr, step)` 推对端 tail
  （`src/device/prims_simple.h:1056-1057`）；
- 接收 kernel 轮询本地 tail、消费后写 head（对发送方可见）释放槽位。

**NET 走 proxy**（`src/transport/net.cc:505-512, 583-590`）：**同一套 connInfo，但
sendMem/recvMem 是本地与 proxy 线程共享的暂存区**（host pinned 或 GDR 内存，connectMap
双端映射）。kernel 原语完全不变，"对端"就是本地 proxy：

发送侧 `sendProxyProgress`（net.cc:1304+）：

1. proxy 先写 `sendMem->head`（或 `gdcSync`）给 GPU 发信用（net.cc:1346-1349）；
2. GPU kernel 把数据写入共享 buff，写 `connFifo[slot].size`（prims_simple.h:386），
   并推 `recvMem->tail`；
3. proxy 轮询 `connFifo[buffSlot].size != -1 && *recvTail > tail`（net.cc:1360-1364）
   后 `isend`；
4. 网络发送 `test` 完成后复位 `connFifo.size = -1`、推进 `done`。

接收侧 `recvProxyProgress`（net.cc:~1500-1765）：

1. proxy `irecv` 到共享 buff（net.cc:1590）；
2. `test` 完成（必要时 `iflush` / GDR flush，net.cc:1641-1686）；
3. 写 `recvMem->tail = base + transmitted` 通知 GPU（net.cc:1709-1713，gdcSync 时
   `wc_store_fence`）；
4. 接收 kernel 消费后推 `sendMem->head`；proxy 轮询 `*sendHead` 确认槽位释放
   （net.cc:1729-1744），`irecvConsumed` 后推进 `done`。

**LL/LL128 的区别**：flag 内嵌在数据线里（`NCCL_LL_FLAG`，device.h:100-106），
不依赖 connFifo 的 size；connFifo 的 size/offset 主要服务 SIMPLE 协议与共享 buffer 模式
（device 侧读写点：prims_simple.h:119/386/1038、prims_ll.h:325-340）。

### 2.4 一次 allreduce 的完整流程

1. host 任务规划 → `uploadWork` 写 workFifo + kernel args → `ncclProxySaveOp` 挂 proxy op 链
   → `cuLaunchKernel` 启动 kernel → `ncclProxyStart` 唤醒 proxy 线程；
2. kernel 各 block 认领 channel，从 workBuf 搬 work 到 smem，按算法（RING/TREE/NVLS…）
   + 协议（LL/LL128/SIMPLE）执行，与邻居 GPU（P2P 直写 tail）或本地 proxy
   （共享 head/tail/connFifo）做信用式握手搬数据；
3. proxy 线程并发推进网络收发，与 GPU 通过 `sendMem->head` / `recvMem->tail` /
   `connFifo.size` 三件套同步；
4. kernel 完成写回 channel `workCounter`；host 用 CUDA event 回调推进
   `workFifoConsumed`，回收 FIFO，开始下一轮。

### 2.5 三层 FIFO 速查表

| FIFO | 位置 | 生产者 → 消费者 | 同步机制 |
|------|------|----------------|----------|
| workFifoBuf | host pinned 或 GDR 显存（init.cc:632） | host 主线程 → device kernel | produced/consumed 游标 + CUDA event 回调回收 |
| proxy ops pool | ncclShm 共享内存（proxy.h:229） | host 主线程 → proxy 服务线程 | mutex + cond；freeOps 原子归还 |
| sendMem/recvMem + connFifo | P2P：对端 GPU 显存；NET：本地与 proxy 共享内存 | device kernel ↔ 对端 GPU / proxy 线程 | head/tail 信用计数 + connFifo.size，sys 作用域存储 |
