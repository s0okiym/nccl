# ib_plugin 数据路径关键函数实现对比：b246b19 vs master

对比对象：`src/ib_plugin.c` 中数据路径核心函数 `ncclIbIsend` / `ncclIbMultiSend` / `ncclIbIrecv` / `ncclIbPostFifo` / `ncclIbIflush` / `ncclIbTest` 及辅助函数 `ncclIbGetRequest` / `ncclIbFreeRequest` / `ncclIbAddEvent`。

- 基线：`b246b19`（2024-06，对应 net v5–v8 API）
- 目标：`master`（2026-04，对应 net v6–v11 API）
- 方法：逐函数提取两个版本的函数体做 diff。

**总体结论先行：两个版本的数据路径算法几乎完全一致**——FIFO/CTS 信用机制、多 QP 条带化、adaptive routing、gpuFlush、完成处理流程在 master 上没有任何重构。全部差异归结为四类：(1) API 类型拓宽（int→size_t、phandle）；(2) 设备计数抽象从 `ndevs` 改为 `vProps.ndevs`/`nDataQps`（v9 虚拟设备）；(3) 新增异步致命错误检查；(4) `ncclIbTest` 错误报告增强与 events 数组扩到 4 槽。

---

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

**一句话总结**：从 b246b19 到 master，send/recv/flush 的传输算法零改动；`ncclIbTest` 是唯一有实质行为变化的函数（异步致命错误即时上抛 + 4 设备支持 + 错误日志可读性）；其余差异全部是 net v9/v10 API 拓宽带来的签名调整和设备计数字段的 vProps 化改名。
