# NCCL IB Send/Recv 全流程：CTS 等待与收发端配合

> 分析对象：`src/transport/net_ib/p2p.cc`、`src/transport/net_ib/connect.cc`、`src/transport/net_ib/common.h`、`src/transport/net_ib/p2p.h`

## 1. 什么是 `ncclIbIsend` 里的 “CTS 等待”

`ncclIbIsend` 在真正发起 RDMA Write 之前，必须等待接收端通过 RDMA 把本次收包的 **Clear-To-Send (CTS)** 信息写到发送端本地的 `ctsFifo` 中。CTS 里携带了接收端已经准备好的目的地址、rkey、tag、期望大小等元数据；发送端只有拿到这些信息后，才知道把数据写到对端哪里。

### 1.1 等待代码位置

`src/transport/net_ib/p2p.cc` 中的 `ncclIbIsend`：

```cpp
int slot = comm->base.fifoHead % NET_IB_MAX_REQUESTS;
struct ncclIbRequest** reqs = comm->sendReqs[slot];
slots = comm->ctsFifo[slot];                       // 发送端本地 CTS FIFO
uint64_t idx = comm->base.fifoHead + 1;
if (slots[0].idx != idx) {                         // ← CTS 等待点
  *request = NULL;
  return ncclSuccess;                              // 还没收到 CTS，下次再试
}
nreqs = slots[0].nreqs;
// 若是 multi-recv，还要等同一 slot 里的其它条目都到齐
for (int r = 1; r < nreqs; r++) {
  while (slots[r].idx != idx);
}
```

- `comm->ctsFifo[slot]` 是一块注册给接收端远程写的内存。
- 接收端在 `ncclIbIrecv` 准备好 buffer 后，会往发送端的这个 slot 写入 `ncclIbSendFifo` 数组，并把 `idx` 设成 `fifoHead + 1`。
- 发送端通过检查 `slots[0].idx == idx` 判断“这一轮的 CTS 已经到了”。

## 2. 接收端在哪里写 CTS 状态

接收端写 CTS 的核心函数是 `ncclIbPostFifo`，位于 `src/transport/net_ib/p2p.cc`：

```cpp
ncclResult_t ncclIbPostFifo(struct ncclIbRecvComm* comm, struct ncclIbRequest* req, int slot) {
  ncclIbQp* ctsQp = NULL;
  NCCLCHECK(ncclIbRecvCommGetQpForCts(comm, req->id, &ctsQp));   // 选第 0 个 QP

  struct ibv_send_wr wr;
  memset(&wr, 0, sizeof(wr));
  // 目标地址：发送端的 ctsFifo 中对应 slot
  wr.wr.rdma.remote_addr = comm->remCtsFifo.addr
                           + slot * NCCL_NET_IB_MAX_RECVS * sizeof(struct ncclIbSendFifo);
  wr.wr.rdma.rkey = comm->base.remDevs[ctsQp->remDevIdx].rkey;
  wr.sg_list = &(comm->devs[ctsQp->devIndex].sge);
  wr.sg_list[0].addr = (uint64_t)localElem;        // 本地准备好的 CTS 条目
  wr.sg_list[0].length = req->nreqs * sizeof(struct ncclIbSendFifo);
  wr.num_sge = 1;
  wr.opcode = IBV_WR_RDMA_WRITE;                   // RDMA Write 到发送端
  wr.send_flags = comm->remCtsFifo.flags;          // 可能 inline
  ...
  NCCLCHECK(wrap_ibv_post_send(ctsQp->qp, &wr, &bad_wr));
}
```

在 `ncclIbIrecv` 里，接收端先把用户 buffer 的信息填充到本地 shadow CTS 条目 `localElem` 中：

```cpp
struct ncclIbSendFifo* localElem = comm->remCtsFifo.elems[slot];
for (int i = 0; i < n; i++) {
  localElem[i].addr = (uint64_t)data[i];
  ...
  for (int j = 0; j < comm->base.vProps.ndevs; j++) {
    localElem[i].rkeys[j] = mhandleWrapper->mrs[j]->rkey;
  }
  localElem[i].nreqs = n;
  localElem[i].size  = sizes[i];
  localElem[i].tag   = tags[i];
  localElem[i].idx   = comm->base.fifoHead + 1;   // 关键：让发送端识别 CTS 已就绪
}
NCCLCHECK(ncclIbPostFifo(comm, req, slot));        // RDMA Write 到发送端
comm->base.fifoHead++;
```

**结论：CTS 是接收端主动通过 RDMA Write 写到发送端 `ctsFifo` 的一小块元数据。**

## 3. 连接建立阶段的关键信息交换

`src/transport/net_ib/connect.cc` 中的 `ncclIbConnect` / `ncclIbAccept` 完成 QP 创建、内存注册和元数据交换。

| 步骤 | 发送端 (`ncclIbConnect`) | 接收端 (`ncclIbAccept`) |
|------|--------------------------|--------------------------|
| 1 | 创建 RC QP（sender QPs），注册本地 `ctsFifo`，得到 `ctsFifoMr->rkey` | 等待并接收发送端元数据 |
| 2 | 通过 socket 发送 `ncclIbConnectionMetadata`：包含 QP 信息、GID、LID、以及 `addr = (uint64_t)ctsFifo`、`rkey = ctsFifoMr->rkey` | 创建 RC QP 并连到发送端 QP |
| 3 | 接收接收端回传的 `ncclIbConnectionMetadata` | 注册本地 `cmplsRecords`，得到 `cmplsRecordsMr->rkey`；注册本地 shadow CTS FIFO (`remCtsFifo.elems`) |
| 4 | 从接收端元数据拿到 `remCmplsRecords.addr` 和各 device 的 `rkey` | 回传 metadata：`addr = (uint64_t)cmplsRecords`、`rkey = cmplsRecordsMr->rkey` |
| 5 | 本地注册 shadow completion records `remCmplsRecords.elems` | 从发送端元数据拿到 `remCtsFifo.addr` 和 `rkey` |

### 3.1 关键内存布局

- **发送端** (`struct ncclIbSendComm`)：
  - `ctsFifo[NET_IB_MAX_REQUESTS][NCCL_NET_IB_MAX_RECVS]` —— 给接收端写 CTS。
  - `remCmplsRecords.elems[slot][r]` —— 发送端本地 shadow，用于把各 sub-req 的大小写回接收端 `cmplsRecords`。
- **接收端** (`struct ncclIbRecvComm`)：
  - `remCtsFifo.elems[...]` —— 本地格式化 CTS 的 shadow。
  - `cmplsRecords[NET_IB_MAX_REQUESTS]` —— 发送端写大小、接收端读大小的 completion records。

## 4. 数据发送与完成的完整流程

### 4.1 接收端先发起动作

1. 用户调用 `ncclIbIrecv`。
2. 接收端分配 `ncclIbRequest`，选一个 slot。
3. 把用户 buffer 地址、`rkey`、`tag`、`size` 等写入 `remCtsFifo.elems[slot][i]`，并设 `idx = fifoHead + 1`。
4. 调用 `ncclIbPostFifo`，通过 **RDMA Write** 把 CTS 写到发送端的 `ctsFifo[slot]`。
5. 接收端递增 `fifoHead`，并可选地 post receive WQE 到数据 QP（取决于 `prepostReceiveWorkRequests`）。

### 4.2 发送端后响应

1. 用户调用 `ncclIbIsend`。
2. 发送端检查 `ctsFifo[slot][0].idx == fifoHead + 1`。
   - 没到 → `*request = NULL`，返回 `ncclSuccess`，下次再试。
   - 到了 → 继续。
3. 检查 tag 是否匹配，size 是否合法。
4. 分配 `ncclIbRequest`，填充 `send.size`、`send.data`、`lkeys` 等。
5. 若 multi-recv，等到 `sendReqsCnt[slot] == nreqs`（即所有 sub-send 都已匹配）后，调用 `ncclIbMultiSend`。
6. `ncclIbMultiSend` 构造 RDMA Write WQE，把数据写到 CTS 中携带的 `slots[r].addr`；并可能额外发起 RDMA Write with IMM，把各 sub-req 大小写到接收端 `cmplsRecords[slot]`。

### 4.3 完成通知

- **发送端**：`ncclIbMultiSend` 的最后一个 WR 带 `IBV_SEND_SIGNALED`。发送端 CQ 产生 CQE，`ncclIbTest` 轮询到后递减 `events[devIndex]`，所有设备事件归零即认为 send 完成。
- **接收端**：
  - 数据到达后，若使用 `BY_ID` matching，带有 immediate 的 RDMA Write 会触发 `IBV_WC_RECV_RDMA_WITH_IMM`，`imm_data` 即 request ID，据此找到对应 `recvReq`。
  - 接收端从 `cmplsRecords[slot].sizes[]` 或 `wc.byte_len` 得到实际收到的 size。
  - 可选的 `ncclIbIflush` 通过 RDMA Read 把数据从 NIC cache 刷到 GPU 可见。

## 5. 一次往返的极简视图

```
接收端                                    发送端
  |                                         |
  |  ncclIbIrecv(data, tag, mhandle)        |
  |  ── 写本地 CTS shadow ─────────────────>|
  |  ── RDMA Write CTS ────────────────────>|  ctsFifo[slot].idx 变有效
  |                                         |
  |                              ncclIbIsend(data, tag, mhandle)
  |                              看到 ctsFifo[slot] 就绪
  |                              取 addr/rkey，RDMA Write 数据
  |<──────────────────────────── RDMA Write data
  |                                         |
  |  收到数据/IMM，写 cmplsRecords          |
  |                              收到 signaled CQE，send 完成
  |  ncclIbTest 返回 done                   |
```

## 6. 核心结论

1. **CTS 等待** = 发送端等接收端把 “我准备好了，数据写这里” 的元数据写到发送端本地 `ctsFifo`。
2. **CTS 写入** = `ncclIbPostFifo` 在接收端被调用，通过 RDMA Write 把 `remCtsFifo.elems[slot]` 的内容写到发送端 `ctsFifo[slot]`。
3. **整体模型** = 接收端驱动发送：先 post receive 并写 CTS，发送端看到 CTS 后再 RDMA Write 数据；完成通过 signaled CQE / completion records 双向确认。
