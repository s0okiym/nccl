# IB 数据路径关键函数实现三方对比：插件 b246b19 vs 插件 master vs NCCL 内部 net_ib

对比对象：数据路径核心函数 `ncclIbIsend` / `ncclIbMultiSend` / `ncclIbIrecv` / `ncclIbPostFifo` / `ncclIbIflush` / `ncclIbTest` 及辅助函数 `ncclIbGetRequest` / `ncclIbFreeRequest` / `ncclIbAddEvent`，来自三个实现：

- **A. 插件 b246b19**：`nccl-rdma-sharp-plugins` 仓库 `src/ib_plugin.c` @ `b246b19`（2024-06，net v5–v8 API，1851 行单文件，C）
- **B. 插件 master**：同文件 @ `master`（2026-04，net v6–v11 API，2335 行单文件，C）
- **C. NCCL 内部实现**：`nccl` 仓库 `src/transport/net_ib/p2p.cc`（2026-07 快照，876 行，C++；为 net_ib 目录重构的一部分，连接建立在 `connect.cc`、初始化在 `init.cc`、弹性恢复在 `p2p_resiliency.cc`、GIN 在 `gin.cc`）

- 方法：逐函数提取三个版本的函数体做 diff。

**总体结论先行**：A→B 数据路径算法几乎完全一致（详见第一部分）；C 是对同一套 FIFO/CTS 协议的**深度重构**——协议骨架（CTS FIFO + RDMA_WRITE 数据 + WRITE_WITH_IMM 完成通知 + 128B 对齐多 QP 分片 + gpuFlush）保持兼容，但请求标识、wr_id 编码、QP 选择、完成检索、多请求 size 回传、AR 触发条件等全部重新设计，并新增接收端匹配方案（BY_ID/BY_INDEX）、弹性恢复（resiliency）、recv WR 预投递、OooRQ 感知与内置 profiling（详见第二部分）。

---

# 第一部分：插件 b246b19（A）vs 插件 master（B）

## 1. ncclIbIsend（发送入口）

**b246b19（ib_plugin.c:1376）→ master（:1724）**，函数体 84→85 行。逐 hunk 差异：

| 位置 | b246b19 | master | 说明 |
|---|---|---|---|
| 签名 | `(sendComm, data, int size, tag, mhandle, request)` | `(sendComm, data, size_t size, tag, mhandle, void* phandle, request)` | net v9 起 size 拓宽为 `size_t`；v10 起新增 `phandle`（设备卸载句柄，**IB 路径不使用**，仅透传给 `ncclIbMultiSend`） |
| 入口 | 无 | `NCCLCHECK(ncclIbStatsCheckFatalCount(&comm->base.stats, __func__))` | **新增**：异步线程上报过 fatal 事件（DEVICE_FATAL/CQ_ERR/QP_FATAL）时直接返回 `ncclSystemError`，不再继续发送 |
| size 不匹配 WARN | `size %d` | `size %ld` | 格式串跟随 size_t |
| events 计数 | `nEvents = SplitDataOnQps ? nqps : base.ndevs` | `... : base.nDataQps` | 见第 7 节 |
| lkeys 存储循环 | `for (i < base.ndevs)` | `for (i < base.vProps.ndevs)` | 结构体字段改名（v9 vProps 化），语义相同 |
| MultiSend 调用 | `ncclIbMultiSend(comm, slot)` | `ncclIbMultiSend(comm, slot, phandle)` | 透传 phandle |

**不变的部分**：`comm->fifo[fifoHead % MAX_REQUESTS]` 轮询等 CTS（`slots[0].idx == fifoHead+1` 后自旋等齐 nreqs、`__sync_synchronize()`）、按 tag 匹配 `fifoReqs` 空槽、size 截断到对端 posted 大小（`slots[r].size`）、凑齐 nreqs 才发送、发送后 `memset` 清槽 + `fifoHead++`。逐行一致。

另外 master 为旧 API 保留收窄包装：`ncclIbIsend_v9`（去 phandle）、`ncclIbIsend_v8`（`int size`→`size_t`）。

## 2. ncclIbMultiSend（WR 链构造与多 QP 分片）

**b246b19（:1272）→ master（:1620）**，函数体 104→104 行，仅两处差异：

```diff
-ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot) {
+ncclResult_t ncclIbMultiSend(struct ncclIbSendComm* comm, int slot,  void* pHandle) {
...
-  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.ndevs;
+  int nqps = ncclParamIbSplitDataOnQps() ? comm->base.nqps : comm->base.nDataQps;
```

`pHandle` 参数在函数体内**完全未被使用**（为 net v10 设备卸载预留的管线）。

**完全不变的核心逻辑**：

- 每个 request 一个非 signaled `IBV_WR_RDMA_WRITE` + 末尾一个 signaled `IBV_WR_RDMA_WRITE_WITH_IMM` 的 WR 链构造；
- `wr_id` 按字节打包最多 8 个 request 索引（`(wr_id >> (j*8)) & 0xff`）；
- 单 request：imm = 实际 size；多 request：最后一个 WR 把各 request 的 size RDMA 写入对端 `remSizesFifo`；
- adaptive routing：`comm->ar && size > NCCL_IB_AR_THRESHOLD(8192)` 时数据走纯 RDMA_WRITE，另发 0 字节 WRITE_WITH_IMM 触发对端完成；
- 多 QP 分片：`DIVUP(DIVUP(size, nqps), 128) * 128` 的 128B 对齐切分（注释仍是为 LL/LL128 协议兼容），rkey 按 `qp->remDevIdx`、lkey 按本地 dev，`qpIndex` 轮询推进。

## 3. ncclIbIrecv（接收入口）

**b246b19（:1540）→ master（:1889）**，47→48 行。差异：

| 位置 | b246b19 | master |
|---|---|---|
| 签名 | `(recvComm, n, data, int* sizes, tags, mhandles, request)` | `(recvComm, n, data, size_t* sizes, tags, mhandles, void** phandles, request)` |
| 入口 | 无 | `NCCLCHECK(ncclIbStatsCheckFatalCount(...))`（同 isend） |
| devBases 填充 | `for (i < base.ndevs)` | `for (i < base.vProps.ndevs)` |
| recv WR 的 QP 数 | `nqps = SplitDataOnQps ? base.nqps : base.ndevs` | `... : base.nDataQps` |

`phandles` 参数同样未被使用。**不变**：每 QP post 空 `ibv_recv_wr`（承接对端 WRITE_WITH_IMM）、每 dev 经 `ncclIbAddEvent` 加事件、随后调 `ncclIbPostFifo` 发 CTS。master 同样保留 `ncclIbIrecv_v9`/`ncclIbIrecv_v8` 收窄包装。

## 4. ncclIbPostFifo（CTS/FIFO 投递）

**b246b19（:1460）→ master（:1809）**，80→80 行。差异仅三处：

- `int* sizes` → `size_t* sizes`；
- CTS QP 选择游标取模：`comm->base.devIndex = (devIndex + 1) % base.ndevs` → `% base.vProps.ndevs`；
- rkeys 填充循环：`j < base.ndevs` → `j < base.vProps.ndevs`。

**完全不变**：`localElem` 填充（addr/size/nreqs/tag/idx + 全部 dev 的 rkey）、RDMA_WRITE 到对端 `remFifo.addr`、**仅当 `slot == ctsQp->devIndex` 时才 signaled** 的 QP 周期性排空机制、`NCCL_IB_USE_INLINE` 时的 `IBV_SEND_INLINE`。

## 5. ncclIbIflush（GDR flush）

**b246b19（:1587）→ master（:1937）**，37→37 行（master 在函数后新增了 `HCA_NAME` 宏定义，供 test 使用）。**唯一差异**：

```diff
-  for (int i = 0; i < comm->base.ndevs; i++) {
+  for (int i = 0; i < comm->base.vProps.ndevs; i++) {
```

签名、只 flush 最后一个非零 size 的 recv、`flushEnabled` 门控、每 dev 在自连接 flush QP 上 post signaled 1 字节 `IBV_WR_RDMA_READ`（把对端 GPU 内存读入 `gpuFlushHostMem` 以强制先前 RDMA write 可见）——全部逐行一致。

## 6. ncclIbTest（完成轮询）——差异最大的函数

**b246b19（:1624）→ master（:1976）**，94→107 行。四处实质差异：

**① 致命错误检查（2 处新增）**

```c
// 函数入口：
NCCLCHECK(ncclIbStatsCheckFatalCount(&r->base->stats, __func__));
// 每个 dev 的 CQ poll 循环末尾：
NCCLCHECK(ncclIbStatsCheckFatalCount(&ncclIbDevs[r->devBases[i]->ibDevN].stats, __func__));
```

实现（master :66）：

```c
static ncclResult_t ncclIbStatsCheckFatalCount(struct ncclIbStats* stat, const char* funcName) {
  if (ncclParamIbAsyncEvents() && __atomic_load_n(&stat->fatalErrorCount, __ATOMIC_RELAXED)) {
    WARN("communicator encountered a fatal error (detected in %s)\n", funcName);
    return ncclSystemError;
  }
  return ncclSuccess;
}
```

含义：b246b19 中异步线程（`ncclIbAsyncThreadMain`）对 `IBV_EVENT_DEVICE_FATAL` 等事件只打 WARN，comm 上的 request 会一直 poll 直到超时；master 中异步线程通过 QP/CQ 的 context 指针（建 QP/CQ 时传入 `&comm->base.stats`）给对应 comm 的 `fatalErrorCount` 计数，`ncclIbTest` 在下一次 poll 时立即上抛 `ncclSystemError`，并停止继续 poll 该设备（注释原文："prevent further polling to reduce error pollution"）。该行为可由新参数 `NCCL_IB_RETURN_ASYNC_EVENTS`（默认 1）关闭。

**② events 数组 2 槽 → 4 槽**

```diff
-    if (r->events[0] == 0 && r->events[1] == 0) {
+    if (r->events[0] == 0 && r->events[1] == 0 && r->events[2] == 0 && r->events[3] == 0) {
```

配合 `NCCL_IB_MAX_DEVS_PER_NIC` 2→4（v9 NIC fusion），`ncclIbRequest.events[]`/`devBases[]`/`lkeys[]` 均扩为 4。

**③ 完成错误 WARN 大幅增强**

```diff
-              localGidStr = inet_ntop(AF_INET6, &r->devBases[i]->gidInfo.localGid, ...);
-              remoteGidStr = inet_ntop(AF_INET6, &r->base->remDevs[i].remoteGid, ...);
-             WARN("NET/IB: Got completion from peer %s with status=%d opcode=%d len=%d vendor err %d (%s)%s%s%s%s hca %s",
+              localGidStr = ibvGetGidStr(&r->devBases[i]->gidInfo.localGid, ...);
+              remoteGidStr = ibvGetGidStr(&r->base->remDevs[i].remoteGid, ...);
+              int reqSize = wc->byte_len;
+              struct ncclIbRequest* req = r->base->reqs+(wc->wr_id & 0xff);
+              if (req && req->type == NCCL_NET_IB_REQ_SEND) {
+                // For Send use the request size as WC byte_len is not reliable
+                reqSize = req->send.size;
+              }
+              WARN("NET/IB: Got completion from peer %s with status=%s(%d) opcode=%s(%d) reqSize=%d vendor_err=%u req_type=%s%s%s%s%s hca %s",
+                  ..., ibvWcStatusStr(wc->status), wc->status, ibvWcOpcodeStr(wc->opcode), wc->opcode, reqSize, ...);
```

- status/opcode 从纯数字改为 `ibvWcStatusStr()`/`ibvWcOpcodeStr()` 字符串 + 数字；
- 发送类错误的 size 改用 `req->send.size`（`wc->byte_len` 对发送不可靠）；
- GID 格式化改用 `ibvGetGidStr()`；
- b246b19 引入的 `hca %s` 保留。

TRACE 行同步改为字符串化输出。

**④ send 完成分支加 NULL 防护**：`if (req->type == NCCL_NET_IB_REQ_SEND)` → `if (req && req->type == ...)`。

**不变**：`wrap_ibv_poll_cq` 每次最多取 4 个 WC、`wr_id & 0xff` 定位 request、组内其余 request 逐字节递减 events、`IBV_WC_RECV_RDMA_WITH_IMM` 的 imm_data 作为单接收 size、多接收从 `sizesFifo` 拷 size、`ncclIbFreeRequest` 归还槽位。

## 7. 辅助函数

- `ncclIbGetRequest` / `ncclIbFreeRequest`（b246b19 :1121/:1139 → master :1457/:1475）：**逐行一致**，无差异。
- `ncclIbAddEvent`（:552 → :730）：**逐行一致**。
- `ncclIbStatsCheckFatalCount` / `ncclIbStatsInit`：master 新增（b246b19 不存在），调用点见上文。

## 8. `ndevs` → `nDataQps` / `vProps.ndevs` 的语义说明

这是所有数据路径函数中反复出现的字段改名，来源于 net v9 的虚拟设备（vFFT/NIC fusion）改造：

- b246b19：`base.ndevs` 表示本 comm 的 merged dev 子设备数（≤2），同时用作"默认数据 QP 数"（未开 `NCCL_IB_SPLIT_DATA_ON_QPS` 时）。
- master：`base.vProps.ndevs` 是本地虚拟设备的子设备数（≤4）；**`base.nDataQps` 是连接建立时计算的 `MAX(本地 vProps.ndevs, 对端 nRemDevs)`**，取代 `ndevs` 作为 isend/irecv/multisend 中默认数据 QP 数。这样在两端 NIC 融合度不一致（如一端 2 rail、一端 4 rail）时仍能正确配对数据 QP，而 `vProps.ndevs` 仅用于本地资源（lkeys、devBases、flush）的遍历。

对**单 rail 或两端对称融合**的部署（`ndevs` 相等），该改动行为等价；只在非对称融合拓扑下才会体现差异。

---

## 差异总览表

| 函数 | 签名变化 | 逻辑变化 | 行为是否等价（对称拓扑） |
|---|---|---|---|
| `ncclIbIsend` | size→size_t，+phandle（未用） | +fatal 检查；ndevs→nDataQps/vProps.ndevs | 等价 |
| `ncclIbMultiSend` | +pHandle（未用） | ndevs→nDataQps；其余逐行一致 | 等价 |
| `ncclIbIrecv` | sizes→size_t*，+phandles（未用） | +fatal 检查；ndevs→nDataQps/vProps.ndevs | 等价 |
| `ncclIbPostFifo` | sizes→size_t* | ndevs→vProps.ndevs（2 处） | 等价 |
| `ncclIbIflush` | 无 | ndevs→vProps.ndevs（1 处） | 等价 |
| `ncclIbTest` | 无 | **+fatal 检查（2 处）、events 2→4 槽、错误 WARN 字符串化、reqSize 修正、NULL 防护** | 完成判定等价；错误处理行为不同（fatal 快速上抛 vs 等超时） |
| `ncclIbGetRequest`/`FreeRequest`/`AddEvent` | 无 | 无（逐行一致） | 等价 |

**一句话总结（A vs B）**：从 b246b19 到 master，send/recv/flush 的传输算法零改动；`ncclIbTest` 是唯一有实质行为变化的函数（异步致命错误即时上抛 + 4 设备支持 + 错误日志可读性）；其余差异全部是 net v9/v10 API 拓宽带来的签名调整和设备计数字段的 vProps 化改名。

---

# 第二部分：NCCL 内部 net_ib/p2p.cc（C）与插件两版（A/B）的重要差异

C 是 NCCL 主仓库对 IB 传输的 C++ 重构（net_ib 目录化：p2p.cc 仅 876 行，连接/初始化/弹性/GIN 拆到独立文件）。**线上协议与 A/B 同源**（CTS FIFO 元素结构、idx 递增、RDMA_WRITE + WRITE_WITH_IMM、128B 对齐分片、gpuFlush 1 字节 RDMA_READ 均保留），但数据路径内部机制有系统性差异。

## 1. 跨函数的全局性差异（最重要的四点）

**① 请求标识与 wr_id 编码全面重做**

| | A/B（插件） | C（net_ib p2p.cc） |
|---|---|---|
| 请求标识 | 请求数组下标（`req - base->reqs`） | 显式 `req->id = fifoHead`（单调递增） |
| 发送 WR 的 wr_id | 每字节打包一个**请求数组下标**：`(reqs[r]-reqs) << (r*8)` | 每字节打包 **FIFO slot**：`(slot & 0xff) << (r*8)` |
| 接收 WR 的 wr_id | 请求数组下标 | **slot** |
| flush WR 的 wr_id | 请求数组下标 | 请求下标 + `NCCL_IB_FLUSH_REQ_WR_ID_OFFSET`（与数据完成区分） |
| 发送侧完成检索 | `reqs[wr_id & 0xff]`，再逐字节解出组内各请求 | `sendReqs[wr_id & 0xff][0]` 得 slot，再遍历 `sendReqs[slot][j]` |
| 接收侧完成检索 | `reqs[wr_id & 0xff]`（与发送共用一张表） | `recvReqs[wc->wr_id]`（按 slot 注册的 recv 请求表）；BY_ID 方案下用 `recvReqs[imm_data % MAX]` |

**② 接收端匹配方案可选：`NCCL_IB_RECEIVER_SIDE_MATCHING_SCHEME`（C 新增）**

- `BY_INDEX`（默认）：与 A/B 一致——单请求时 imm 携带 size，多请求时 size 由发送端 RDMA 写入接收侧的 per-slot 数组。
- `BY_ID`（C 独有）：imm 携带**请求 ID**（`reqs[0]->id % UINT32_MAX`），接收端按 ID 查 `recvReqs` 表，size 用 `wc->byte_len` 累加到 `req->recv.aggSize`。

**③ Resiliency（弹性/故障恢复）子系统——C 独有，贯穿所有数据路径函数**

- `p2p_resiliency.cc`/`p2p_resiliency_recovery.cc`（约 12 万字节）实现 QP 故障后的连接重建；A/B 完全没有对应物（B 只有 fatal 计数上抛）。
- `ncclIbMultiSend`：**选择性重传**——`req->send.sentData[qpIndex]` 记录每个 QP 是否已送达，重发时跳过已送达 QP 并前移地址（p2p.cc:161-170）。
- `ncclIbTest`：入口先跑 `ncclIbResiliencyProgress()`；完成错误不再直接返回 `ncclRemoteError`，而是交给 `ncclIbResiliencyHandleCompletionError()` 尝试恢复；容忍 events 变负、容忍检索到 unused 请求（INFO 后跳过）。
- `ncclIbIrecv`：resiliency 下预先给所有 dev 填 `devBases`（"a recv request can be served by any device"）。
- `ncclIbPostFifo`：resiliency 下 CTS **总是 signaled**（A/B 仅 `slot == ctsQp->devIndex` 时）。

**④ QP 选择策略：确定性按请求 ID 映射 vs 轮询游标**

- A/B：发送/接收各维护轮询游标（`base.qpIndex`、`base.devIndex`），每次 +1 取模。
- C：`ncclIbCommBaseGetQpForRequest(base, req->id, i, &qp, &qpIndex)`（isend/irecv/multisend 统一按请求 ID 确定性选 QP）；CTS QP 用 `ncclIbRecvCommGetQpForCts(comm, req->id)`。配合 resiliency 可以在设备不可用时换 QP 重发。

## 2. 逐函数差异

### ncclIbIsend

| 方面 | A/B | C |
|---|---|---|
| FIFO 等 CTS | `slots[0].idx == fifoHead+1`，自旋等齐 nreqs | **相同**（`std::atomic_thread_fence(seq_cst)` 替代 `__sync_synchronize()`） |
| tag 匹配 / size 截断 / sanity WARN | 有 | **相同** |
| slot 状态管理 | `fifoReqs[slot]` 匹配即占用；发完立即 `memset` 清槽 | `sendReqs[slot][]` + `sendReqsCnt[slot]` 计数；**延迟到 `ncclIbRequestComplete` 中计数归零才清槽**（防止多请求组内早清） |
| events 填充 | 按 nEvents 轮询 QP `ncclIbAddEvent` | 按 `GetQpForRequest(req->id, i)` 选 QP 加 event；语义相同 |
| sizes 源缓冲 | 无（在 MultiSend 内按需填 `remSizesFifo.elems`） | isend 中无条件 `remCmplsRecords.elems[slot][r] = size` |
| profiling | B 接受 `phandle` 但**不用** | `NCCL_ENABLE_NET_PROFILING` 下记录 `pHandle`、每 QP 起 `ncclProfilerNetEventStart` |
| resiliency | 无 | `sentData` 清零初始化 |

### ncclIbMultiSend

| 方面 | A/B | C |
|---|---|---|
| immData | 单请求=size；多请求=0/1 标志 | BY_INDEX 同 A/B；**BY_ID=请求 ID** |
| 多请求 size 回传目标 | `remSizesFifo`（每 slot 一个 int 数组） | `remCmplsRecords`（每 slot 一个 `ncclIbRequestCompletionRecord`，含 sizes[] + completions[]） |
| AR 触发条件 | `comm->ar && size > IB_AR_THRESHOLD` | **额外条件 `!(remOooRq && localOooRq)`**：两端 RQ 都支持乱序时无需 0 字节 imm 尾部触发完成 |
| 分片偏移跟踪 | `req->send.offset` 累积（含空片也加 chunkSize），`MIN(size-offset, chunk)` | 局部 `sendOffsets[]` + `std::min(size-offset, length)`；**并记录 `sentData[qpIndex]`** |
| 空片处理 | `sg_list=NULL, num_sge=0` | `num_sge=0`（等价） |
| QP 迭代 | `qpIndex=(qpIndex+1)%nqps` 轮询 | `GetQpForRequest(req->id, i)` 确定性映射 |
| 重传 | 无 | resiliency 下跳过已送达 QP |
| 尾部 imm WR | 条件相同（多请求或 AR 大消息） | 相同；BY_ID 时 imm 为 ID |
| profiling / TRACE | 无 / 少量 | 每 QP profiling 事件 + `ncclIbPrintWr` 完整 WR 链转储 |

### ncclIbIrecv

| 方面 | A/B | C |
|---|---|---|
| recv WR 投递 | 每次 irecv 在所有数据 QP 上 post 空 recv WR | **支持 `prepostReceiveWorkRequests`**：预投递后 irecv 不再 post，完成时在 `ncclIbCompletionEventProcess` 中按 `wc->qp_num` 找到原 QP 补投 |
| 请求登记 | 无表 | `recvReqs[req->id % MAX] = req`（供完成检索） |
| 完成记录 | 依赖对端写 `sizesFifo[slot]` | 本地 `cmplsRecords[slot]`（sizes + per-QP completions 标志），irecv 时清零 |
| CTS QP 选择 | `base.devIndex` 轮询游标 | `ncclIbRecvCommGetQpForCts(comm, req->id)` 按 ID 选择 |
| localElem 填充 | addr/rkeys/nreqs/size/tag/idx | **完全相同** |
| fifoHead++ 时机 | PostFifo 之后 | 相同 |

### ncclIbPostFifo

| 方面 | A/B | C |
|---|---|---|
| signaled 条件 | `slot == ctsQp->devIndex`（QP 周期排空机制，注释同源） | 相同条件 **或 resiliency 时总是 signaled** |
| signaled 完成是否计入 recv 请求 events | **是**——`ncclIbAddEvent(req, ctsQp->devIndex)`，recv 请求要等自己的 CTS 排空才算完成 | **否**——CTS 完成在 test 中作为 `IBV_WC_RDMA_WRITE` 分支仅 TRACE 忽略 |
| rkey | `remDevs[remDevIdx].fifoRkey` | `remDevs[remDevIdx].rkey`（字段改名） |

### ncclIbIflush

三者几乎一致：只对最后一个非零 size 的 recv flush、每 dev 在自连接 gpuFlush QP 上 post signaled 1 字节 `IBV_WR_RDMA_READ`、逐 dev `ncclIbAddEvent`。C 的唯一区别是 wr_id 加 `NCCL_IB_FLUSH_REQ_WR_ID_OFFSET`，使 test 能靠 `(opcode==IBV_WC_RDMA_READ, wr_id-offset)` 明确识别 flush 完成（A/B 靠"recv 侧的 RDMA_READ 完成即 flush"这一隐式约定）。

### ncclIbTest

| 方面 | A/B | C |
|---|---|---|
| 结构 | 单体内联（A 94 行 / B 107 行） | 重构为 5 个 helper：`RetrieveFromCompletion` / `RequestIsComplete` / `RequestComplete` / `LogCompletionWithError` / `CompletionEventProcess` |
| 轮询框架 | `while(1){...; if (totalWrDone==0) return;}` | `do{...}while(totalWrDone>0)`——**行为等价**（都是榨干当前可用 CQE 后返回） |
| 完成判定 | events[0..3] 全 0 | 相同；resiliency 下另调 `ncclIbResiliencyRequestIsComplete` |
| dev 轮询跳过条件 | `r->events[i]`（隐式依赖 devBases 非空） | `!r->devBases[i] \|\| (events[i]==0 && !resiliency)`——更显式，容忍发送端未用全部设备 |
| 完成错误 | 返回 `ncclRemoteError` | resiliency 关闭时相同；**开启时进入恢复流程**；另加 `printIbWcStatusHint()` 给出排查提示 |
| recv size 上报 | 单请求取 imm、多请求取 `sizesFifo` | BY_INDEX 相同；BY_ID 用 `aggSize`（byte_len 累加）；多请求或 sizes[0]>0 时报 `cmplsRecords->sizes` |
| send 完成收尾 | `ncclIbFreeRequest` | 额外 `sendReqsCnt[slot]--`，归零才清 `sendReqs[slot]`（与 isend 的延迟清槽配合） |
| CTS / 未知完成 | 走通用 events 递减路径 | 显式分支：CTS 完成忽略、未知 opcode WARN + `ncclInternalError` |
| fatal 检查 | B 有（2 处），A 无 | 与 B 相同（2 处 `ncclIbStatsCheckFatalCount`） |
| profiling | 无 | 完成时逐 QP `ncclProfilerNetEventStop` |

### 辅助函数

- `ncclIbGetRequest`：三者逻辑一致（找 UNUSED 槽）；C 额外清零 `devBases`/`events` 并置 `sock=NULL`。
- `ncclIbFreeRequest` / `ncclIbAddEvent`：三者一致（C 的 AddEvent 经 `ncclIbGetNetCommDevBase()` 取 devBase，多一层间接）。

## 2.5 协议兼容性要点

C 与 A/B 的 CTS FIFO 元素（`ncclIbSendFifo`：addr/size/rkeys/nreqs/tag/idx）、idx 握手、128B 对齐分片、gpuFlush 机制保持同构，但 **wr_id 编码、imm 语义（BY_ID）、多请求 size 回传布局（remSizesFifo vs remCmplsRecords）、recv WR 预投递**均不同——这些属于通信双方内部约定，两端必须同版本；net_ib 作为 NCCL 内置传输天然两端一致，而插件协议与 NCCL 内部插件协议各自成体系，不可混用。

## 三方差异总览表

| 维度 | A. 插件 b246b19 | B. 插件 master | C. NCCL net_ib p2p.cc |
|---|---|---|---|
| 语言/组织 | C，单文件 | C，单文件 | C++，net_ib 目录（p2p/connect/init/gin/resiliency 拆分） |
| 传输算法骨架 | FIFO/CTS + RDMA_WRITE + 128B 分片 | 同 A（逐行级一致） | 同构但全面重构 |
| 请求标识 | 数组下标 | 同 A | 显式 `req->id` + slot 表（sendReqs/recvReqs） |
| 接收匹配方案 | 仅 BY_INDEX | 仅 BY_INDEX | BY_INDEX（默认）/ **BY_ID** 可选 |
| QP 选择 | 轮询游标 | 同 A | 按请求 ID 确定性映射 |
| 多设备支持 | ≤2 | ≤4（vProps/nDataQps） | ≤4（vProps）+ 非对称拓扑 |
| fatal 错误上抛 | 无 | 有（stats 计数） | 有（同 B） |
| **故障恢复** | 无 | 无 | **resiliency：QP 重建 + 选择性重传 + 错误容忍** |
| AR 触发 | size>阈值 | 同 A | 额外要求非 OooRQ 双端 |
| recv WR 预投递 | 无 | 无 | **有（可选）**，完成时按 qp_num 补投 |
| CTS 完成计入 recv events | 是 | 是 | 否（显式忽略分支） |
| profiling | 无 | 接口接受但不用 | **内置**（每 QP start/stop 事件） |
| GIN（GPU 发起） | 无 | 头文件预留，未实现 | **有（gin.cc，IPut 请求类型）** |
| 错误日志 | 数字 status | 字符串化 + reqSize + HCA | 字符串化 + HCA + `printIbWcStatusHint` |

**一句话总结（三方）**：A→B 是"同一实现的小步演进"（API 拓宽 + fatal 检查 + 4 设备）；B→C 是"同一协议的重新实现"——传输骨架不变，但请求/wr_id 标识体系、QP 选择、完成检索全部重设计，并引入 resiliency 故障恢复、BY_ID 匹配、recv 预投递和内置 profiling 四个 A/B 完全没有的能力。

---

# 第三部分：性能分析——net_ib 对 NUMA balancing / runqueue delay 敏感而插件 b246b19 免疫的根因

背景现象：nccl-tests（all_reduce_perf，2 机 16 卡，32MB buffer）下，C（NCCL 内部 net_ib）在 `ncclIbMultiSend` 构造 WR 及 `ibv_post_send` 时出现毫秒级延迟毛刺，整体延迟 ~500µs 抖动，且对 `kernel.numa_balancing` 开关和 runqueue delay 类调度参数非常敏感；A（插件 b246b19）在相同环境延迟稳定 ~260µs，对两者免疫。

## 1. 结论

**最可能的原因：C 的热路径上"未注册、可被内核迁移"的簿记内存工作集从 A 的几页膨胀到几十~几百页，且按 `id % 256` 旋转索引访问，使 NUMA balancing 的 hinting fault 扫描 + 页迁移在每次发送中高频命中；同时 C 单次 proxy progress 调用的连续 CPU 时间更长，proxy 线程更易被调度器迁移（runqueue delay 敏感点）。A 的热路径只触碰 2~3 页未固定内存，预热后稳定驻留本地节点，故免疫。**

毛刺的本质不是代码阻塞，而是内核对该线程的页（hinting fault → 页迁移，单次数百 µs~数 ms，THP 拆分/规整时更糟）和线程本身（跨 NUMA 迁移）做动作；代码差异决定的是暴露面大小。

## 2. 机制链

1. `kernel.numa_balancing=1`：内核周期性 unmap 进程用户态页做 NUMA 扫描；线程再访问该页触发 hinting fault，可能伴随页迁移，迁移期间线程睡眠数百 µs~数 ms。
2. 关键前提：被 `ibv_reg_mr` 注册过的页由 RDMA 驱动 pin 住，**NUMA balancing 无法迁移**——CTS FIFO、远端 size 回传区、gpuFlush host 内存在三个实现中均已注册，不是差异来源。暴露在迁移风险下的是**未注册的簿记内存**。
3. 每次 isend/irecv/test 必触碰的未固定内存：

| 结构 | A. 插件 b246b19 | C. NCCL net_ib |
|---|---|---|
| 请求池 | `reqs[64]` × ~64B ≈ 4KB | `reqs[256]` × ~250B（含 `sentData[128]`）≈ **64KB**；若编译开 `NCCL_ENABLE_NET_PROFILING`，`pInfo[8]`/request 可再膨胀至 **~1MB** |
| slot→请求映射 | `fifoReqs[64][8]` = 4KB | `sendReqs[256][8]` = 16KB + `sendReqsCnt[256]` + `recvReqs[256]` |
| 访问模式 | slot 紧凑复用，热页 **2~3 页** | `id % 256` **旋转索引**，连续发送跨整个数组跨页 stride 访问 |
| 每次 isend 额外写 | 无 | `sentData` memset、`pInfo[0].nEventHandles=0`；irecv 还有 `cmplsRecords` 双 memset |
| **未固定热工作集合计** | **~8KB（几页）** | **~100KB–1MB+（几十~几百页）** |

  （`NET_IB_MAX_REQUESTS = NCCL_NET_MAX_REQUESTS(32) × NCCL_NET_IB_MAX_RECVS(8) = 256`；插件为 `8 × 8 = 64`。）

4. C 每次发送写分布在几十~几百页中的不同 slot，numa balancing 扫描周期内几乎必然踩到刚被 unmap 的页 → hinting fault → 页迁移，线程睡在 MultiSend 构 WR 或紧随的 `ibv_post_send`（写 QP 驱动缓冲 + doorbell MMIO，同样受线程/页位置影响）中——与观测的毛刺位置吻合。
5. C 的 proxy 线程单次 progress 调用连续 CPU 时间更长（见第 3 节加重因素），更易被调度器判为可迁移的 CPU hog；线程一旦跨节点迁移，后续 comm 内存与 HCA doorbell 全部变跨节点访问——解释对 runqueue delay 的敏感性。
6. A 的热工作集只有 2~3 页，预热后稳定驻留 proxy 线程本地节点，扫描命中概率极低 → 对两个开关都不敏感。500µs vs 260µs 的均值差 = 高频小迁移 + 偶发大迁移的叠加。

## 3. 加重因素（C 独有）

- `ncclIbTest` 榨干循环内多做：按 `wc->qp_num` 线性反查 QP（`ncclIbCommBaseGetQpByQpNum`）、补投预投递的 recv WR、profiling stop 事件；
- isend/irecv/multisend 的 profiling 簿记写入（即使未挂 profiler 插件，`pInfo` 字段写仍执行；插件 master 接受 `profFunction` 但从不使用，零热路径成本）；
- resiliency 字段维护（`sentData` memset 等），即使该功能默认关闭。

## 4. 已排除的嫌疑

- `wrap_ibv_post_send`：A/C 均为裸调 `qp->context->ops.post_send`，无锁；
- 内存分配：两边 `ncclIbMalloc` 都是 page 对齐 `posix_memalign`，无差别；
- CTS 自旋等待、完成榨干循环框架：等价；
- QP 选择（`(id*nQps)%nqps` vs 轮询游标）：常数级差异，与 ms 毛刺无关。

## 5. 验证方法

1. 跑 all_reduce_perf 时对比两种实现 `/proc/vmstat` 的 `numa_hint_faults`、`numa_pages_migrated`、`numa_pte_updates` 增量——C 应显著非零，A 接近零（最直接证据）。
2. `perf record -e page-faults -p <proxy线程>` 或 `perf sched latency`，确认毛刺时刻对应 fault/migration。
3. 确认 NCCL 构建是否定义 `NCCL_ENABLE_NET_PROFILING`（`nm` 查 `ncclProfilerFunction` 是否被链入 so）——若是，`reqs[]` 还要大一个量级。
4. 反向验证：对 C 的 comm 簿记内存 `mlock`，毛刺应消失。

## 6. 缓解方向

- 短期（已验证有效）：关 `kernel.numa_balancing`；proxy 线程绑定到 HCA 所在 NUMA 节点（numactl / NCCL CPU affinity）。
- 代码层（针对根因）：`reqs[]`/`sendReqs`/`recvReqs` 等未注册簿记内存 `mlock` 或绑节点分配（`mbind`/`MADV_HUGEPAGE`）；profiling 簿记改为"无 profiler 插件时完全跳过"；缩小 `NET_IB_MAX_REQUESTS` 或紧凑化 slot 复用，削减热工作集。

## 7. 补充：关闭 numa_balancing 后的残余毛刺——CPU 调度（runqueue delay）敏感性分析

现象：关闭 `kernel.numa_balancing` 后 C 的毛刺缓解 >99% 但不归零，仍对 CPU 调度（runqueue delay 类参数）敏感；A 不受影响。

### 结论

**残余毛刺来自"全自旋、零让出"的数据路径被调度器抢占/迁移——这是 A、C 共有机制，故无法归零；C 更敏感是因为每次 net 回调的 CPU 工作量更大（暴露窗口更长）且热工作集更大（被抢占/迁移后的恢复成本更高）。**

### 机制链

1. 整条延迟关键路径没有任何让出 CPU 的点（A、C 共有）：proxy 服务线程 busy-poll（`proxy.cc` 主循环仅在"本次循环零推进"时才 `std::this_thread::yield()`，稳态下几乎不 yield）→ `ncclIbIsend` 中 `while (slots[r].idx != idx);` 自旋等 CTS → `ncclIbMultiSend` 逐 QP 构 WR + `ibv_post_send` → `ncclIbTest` 的 `do/while` 榨干 CQE。**任何调度事件（timer tick、同核 kworker/IRQ/CUDA 线程唤醒抢占、线程迁移）的延迟都 1:1 注入集合通信关键路径**——残余毛刺无法靠关 numa balancing 归零的原因。
2. proxy 线程为 SCHED_OTHER、无优先级/隔离保护，被抢占后需等 runqueue 重新调度（`runqueue_delay`/migration cost 类参数的调节点）；若被迁移则缓存全冷。
3. C 比 A 敏感的代码级原因——暴露窗口与单次代价：

| 因素 | A. b246b19 | C. net_ib | 调度敏感性后果 |
|---|---|---|---|
| 每次 net 回调指令量 | 小：紧凑 request（~64B）、少量簿记 | 大：`sentData[128]` memset、`pInfo` 簿记、`cmplsRecords` 双 memset、`CompletionEventProcess` 多分支、按 `wc->qp_num` 线性反查 QP（`ncclIbCommBaseGetQpByQpNum`）、多层间接调用 | 窗口越长，抢占落在 isend→post_send→test 链中间的概率越高 |
| 被迁移后恢复成本 | 热工作集 ~8KB，L1/L2 常驻 | ~100KB–1MB+，且 `id%256` 旋转访问，未迁移时也持续 L2/L3 miss | 每次迁移需重拉取大工作集，慢尾 10~100 倍于 A |
| 完成处理 | 单体内联 | 每 CQE 走 helper 链 + 可选补投 recv WR + profiling stop 间接调用 | 榨干循环在高负载下单次驻留更久 |

4. 32MB all_reduce（2 机 16 卡）是大量串行 net op 组成的长链，链上任一处 50~500µs 抢占即表现为端到端抖动。A 并非不被抢占，而是窗口短、恢复代价小，统计上"免疫"。

### 已排除

- resiliency / OooRQ / recv 预投递：默认全关（`IB_RESILIENCY_PORT_FAILOVER=0`、`IB_OOO_RQ=0`、`IB_PREPOST_RECEIVE_WORK_REQUESTS=-2→false`），无恢复线程/心跳流量；
- isend 的 CTS 自旋逻辑：A、C 逐行一致；
- async event 线程：两边均每 IB 设备一个、阻塞于 `ibv_get_async_event`，不耗 CPU；
- `ibv_post_send` 无锁。

### 验证

- 测试期间对 proxy 线程 `perf sched record` → `perf sched latency` + migrations 统计，A/C 对比：预期抢占次数相近，但 C 单次 off-CPU 影响更大；`/proc/<pid>/sched` 看 `nr_migrations`/`nr_involuntary_switches`。
- 决定性实验：proxy 线程绑隔离核（`isolcpus`/`nohz_full` + IRQ 移离）或 `chrt -f` 提 SCHED_FIFO——C 的残余毛刺收敛到 A 的水平即坐实调度源。

### 缓解

- 系统层：proxy 线程 CPU 隔离 + 绑核（HCA 所在 NUMA）、SCHED_FIFO、IRQ affinity 移离 proxy 核。
- 代码层：与本部分第 6 节同源——裁剪每 op 簿记（无 profiler 时跳过 pInfo 写）、缩小请求/slot 数组、簿记内存 `mlock`，同时降低页迁移暴露（第 2~6 节）与调度暴露窗口（本节）。
