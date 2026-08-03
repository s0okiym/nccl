# ib_plugin 代码分析：commit b246b19 实现解析与 master 对比

- 分析对象：`src/ib_plugin.c`（InfiniBand/RoCE verbs 传输后端，NCCL 网络插件的核心文件）
- 基线 commit：`b246b19c264fc1a673db098cc2535a0d1a0a10fd`（"update"，Devendar Bureddy，2024-06-07）
- 对比目标：`master`（截至 2026-04，b246b19 是其祖先，中间相隔 33 个 commit，其中 10 个直接改动了 `ib_plugin.c`）
- 文件规模：b246b19 版 1851 行 → master 版 2335 行（+484 行；伴随 `p2p_plugin.c` 557→786、`ibvwrap.c` 209→345、`p2p_plugin.h` 147→176）

---

## 第一部分：b246b19 版本 ib_plugin.c 实现分析

### 1. 角色与插件 API 接口

`ib_plugin.c` 是 P2P RDMA 传输的默认后端（`NCCL_PLUGIN_P2P=ib`）。它不直接导出符号，而是定义 4 个版本化 net API 结构体，由 `p2p_plugin.c` 的 `pluginSetup()` 拷贝进导出的 `ncclNetPlugin_v5…v8`：

| 结构体 | 名称 | 说明 |
|---|---|---|
| `ibPlugin_v8` | `"IBext_v8"` | 完整 v8 API，含 `regMrDmaBuf`；`getDeviceMr`/`irecvConsumed` 为 NULL |
| `ibPlugin_v7` / `v6` / `v5` | `"IBext_v7/6/5"` | 递减的旧版 API |

实现的回调（均以 `ncclIb` 开头）：`ncclIbInit`、`ncclIbDevices`、`ncclIbGetProperties`、`ncclIbListen`、`ncclIbConnect`、`ncclIbAccept`、`ncclIbRegMr`/`ncclIbRegMrDmaBuf`、`ncclIbDeregMr`、`ncclIbIsend`、`ncclIbIrecv`、`ncclIbIflush`、`ncclIbTest`、`ncclIbCloseSend/Recv/Listen`。

此版本**没有** `getDeviceMr`/`irecvConsumed` 实现，**没有** v9+ API，也**没有** 向 NCCL 上报 fatal 异步错误的机制——完成错误只能通过 `ncclIbTest` 返回 `ncclRemoteError` 传播。

### 2. 关键数据结构

常量：`NCCL_NET_IB_MAX_RECVS=8`（一次 irecv 最多 8 个 buffer）、`MAX_REQUESTS=64`、`NCCL_IB_MAX_DEVS_PER_NIC=2`、`NCCL_IB_MAX_QPS=128`、`MAX_IB_DEVS=32`。

- `ncclIbDev`（p2p_plugin.h）：一个 IB 设备:端口一份——`guid`、`portNum`、`portAttr`、共享且引用计数的 `pd`（`pdRefs`）、`pciPath`、`speed`、`maxQp`、`mrCache`、`ar`（adaptive routing）、`isSharpDev`、per-dev 锁。
- `ncclIbMergedDev`：一个"NCCL 设备"最多融合 2 个 `ncclIbDev`（多端口 NIC 合并），聚合 speed，devName 用 `+` 连接。
- `ncclIbDevInfo`（ib_plugin.c:371）：连接时交换的 per-dev 线格式元数据——`lid`、`ib_port`、`mtu`、`link_layer`、`is_global`、`union ibv_gid gid`（**b246b19 把旧的 `spn`/`iid` 两个 uint64 换成完整 GID**，以便携带 FLID）、`fifoRkey`、`remoteGid`。
- `ncclIbConnectionMetadata`（:390）：唯一的 OOB 交换结构（收发双向共用），含 `qpInfo[128]`（qpn + ECE）、`devs[2]`、`fifoAddr`、`ndevs`；最终握手是一个裸 `int ready`。
- 连接状态机：`enum ncclIbCommState`（Start/Connect/Accept/Send/Recv/Connecting/Connected/PendingReady）+ `ncclIbCommStage`，配合 goto 标签实现**可恢复的非阻塞 connect/accept**。
- `ncclIbSendFifo`（:437）：64 字节 CTS 元素（静态断言保证 32B 对齐，防止 relaxed ordering 拆散单个元素）：`addr`、`size`、`rkeys[2]`、`nreqs`、`tag`、`idx`。
- `ncclIbNetCommBase`：`reqs[64]`、`qps[128]`、`nqps`、`qpIndex`（轮询游标）、`remDevs[2]`。
- `ncclIbSendComm`：`fifo[64][8]`（接收端 RDMA 写入 CTS 的目标）、`fifoReqs`、`remSizesFifo`（多请求时回传每个 request 的实际 size）、`fifoHead`、`ar`。
- `ncclIbRecvComm`：`remFifo`（本地暂存 CTS 元素 + 对端地址）、`sizesFifo[64][8]`、`gpuFlushHostMem`、每 dev 一个**自连接的 gpuFlush QP**、`flushEnabled`。
- MR cache：页粒度、按地址排序的 slot 数组，引用计数，容量从 32 翻倍增长；挂在 `ncclIbDev` 上，由 `ibDev->lock` 保护。`ncclIbMrHandle` 内含每个 merged dev 一个 `ibv_mr*`（`mrs[2]`）。

**多 QP 条带化**：`nqps = NCCL_IB_QPS_PER_CONNECTION(默认1) * ndevs`，QP 在 merged devs 间轮转创建；数据按 QP 分片（见第 6 节）。

### 3. 设备发现与初始化（ncclIbInit）

1. 检查 `NCCL_IBEXT_DISABLE`；委托 `nccl_p2p_ib_init()`（p2p_plugin.c），由 `nccl_p2p_lock` 保护且只执行一次。
2. OOB bootstrap 网口选择（`ncclFindInterfaces`）；解析 `NCCL_IB_HCA`（`^` 黑名单、`=` 精确匹配）。
3. `ibv_get_device_list` → 逐设备 `ibv_open_device`/`ibv_query_device` → 逐端口 `ibv_query_port`，只保留 `IBV_PORT_ACTIVE` 且链路层为 IB 或 Ethernet 的端口。
4. 填充 `ncclIbDevs[]`：`pciPath` 来自 `/sys/class/infiniband/<dev>/device` 的 realpath（端口号位清零以合并同卡端口；`NCCL_IB_MERGE_VFS=1` 时 VF 位清零）；speed = `active_speed * active_width`。
5. adaptive routing：IB 链路默认开、RoCE 默认关，`NCCL_IB_ADAPTIVE_ROUTING`（默认 -2 未设置）可覆盖。
6. IB 链路端口标记 `isSharpDev=1`，设备排序 SHARP 优先；每个设备起一个 detached 异步事件线程。
7. `ncclIbFindMatchingDev()` 合并同 pciPath+guid+链路层的端口（≤2）。**b246b19 新增**：若出现单端口/多端口 NIC 混杂导致 merged dev 的 ndevs 不一致，打印日志并强制关闭 `NCCL_IB_MERGE_NICS` 重建列表（p2p_plugin.c 的 `goto build_ib_list`）。
8. `ncclIbRelaxedOrderingEnabled` 由能力检测得出（`NCCL_IB_PCI_RELAXED_ORDERING`：1=强制，2=自动）。

`ncclIbGetProperties`：`ptrSupport = HOST | (CUDA if GDR) | (DMABUF if dmabuf probe 成功)`，`regIsGlobal=1`，`maxRecvs=8`，`netDeviceType=HOST`。DMA-BUF 探测用 `fd=-1` 的 dummy `ibv_reg_dmabuf_mr` 调用，靠 errno 区分支持与否。

GID 选择辅助：`NCCL_IB_ADDR_FAMILY`/`NCCL_IB_ADDR_RANGE` 环境变量、RoCE 版本自动探测（sysfs `gid_attrs/types`）、`ncclUpdateGidIndex()`。

### 4. 连接建立（listen/connect/accept）

双方通过 OOB TCP socket 运行可恢复状态机，线协议为三个消息：**sender metadata → receiver metadata → ready int**（均经 `ncclSocketProgress` 分块收发）。

- **ncclIbConnect（发送侧，:696）**：分配 `ncclIbSendComm` → 异步 socket 连接 → 每 dev `ncclIbInitCommDevBase`（共享 PD + CQ，CQ 深度 `2*MAX_REQUESTS*qpsPerConn`）→ 跨 dev 轮转创建 RC QP（`IBV_ACCESS_REMOTE_WRITE`）并查询 ECE → 注册 send FIFO MR（`LOCAL_WRITE|REMOTE_WRITE|REMOTE_READ`）→ 填充 devInfo（含 GID 选择，b246b19 起对**所有**链路类型都执行）→ 互发 metadata、校验 ndevs 与链路层一致性 → 注册 remSizesFifo MR → 逐 QP `set_ece` → `ncclIbRtrQp(..., override_tc=false)` → `ncclIbRtsQp` → 发送 ready。
- **ncclIbAccept（接收侧，:912）**：接受 socket → 收 sender metadata → 建 QP；若对端支持 ECE 则 `set_ece` 后重新查询"削减后"的 ECE 回传 → **`override_tc = (q == 0)`**（b246b19 的 FIFO 流量类别覆盖）→ RTR/RTS → `flushEnabled = (GDR || DMABUF) && !NCCL_GDR_FLUSH_DISABLE` → 注册 remFifo 暂存 MR、可选的 gpuFlush host MR + 自连接 flush QP、注册 sizesFifo MR（其 rkey 即 `fifoRkey`）→ MTU 取 `MIN(本地, 对端)` → 回发 metadata、等 ready。

### 5. QP 管理与 b246b19 引入的特性

- `ncclIbCreateQp`（:593）：`IBV_QPT_RC`，`max_send_wr=2*MAX_REQUESTS`、`max_recv_wr=MAX_REQUESTS`，inline data = `sizeof(ncclIbSendFifo)`（仅当 `NCCL_IB_USE_INLINE`）。
- `ncclIbRtrQp`（:616）——b246b19 新签名：`(qp, ncclIbGidInfo* sGidInfo, dest_qp_num, ncclIbDevInfo* info, bool override_tc)`：
  - **RoCE**：始终带 GRH，dgid 取自 `info->gid`，`sgid_index` 为本地 GID index，hop_limit=255；`traffic_class` 当 `override_tc` 且 `NCCL_IB_FIFO_TC` 非 0 时用 FIFO TC，否则用 `NCCL_IB_TC`。
  - **IB**：`same_subnet` = 本地/对端 GID 的子网前缀低 16 位比较。默认同子网：`is_global=0`、`dlid=info->lid`。跨子网（或 `is_global`）：dlid 改用对端 **FLID**（从 `info->gid` 提取；FLID==0 时 WARN 并回退 lid），并挂 GRH。
- `ncclIbRtsQp`（:667）：timeout=18、retry_cnt=7、rnr_retry=7。

**b246b19 这个 commit 本身引入的三大特性**：

1. **IB 多子网路由（FLID）**：
   - GID 布局注释（:254）：raw GID = `10b fixed | 22b 0 | 16b FLID | 16b subnet-prefix | 64b EUI`。
   - `ncclIbExtractLocalSubnetPrefix()` / `ncclIbExtractFlid()`（读 gid->raw 第 4-5 字节）。
   - `ncclIbGetGidIndex()` 改签名接收 `ibv_port_attr*`；IB 链路下探测 GID index `NCCL_IB_ROUTABLE_FLID_GID_INDEX`（默认 1），其 FLID ≠ 0 则选用（可路由 FLID），否则回退 GID 0；RoCE 路径逻辑不变。
   - `ncclIbDevInfo.spniid` → `union ibv_gid gid`，使 FLID 随连接元数据传输；`ncclIbRtrQp` 中新增 same-subnet 判断与 FLID-as-dlid 分支。
2. **`NCCL_IB_FIFO_TC`（默认 0）/ override_tc**：仅接收侧 QP 0 传 `override_tc=true`——该 QP 承载 FIFO/CTS 控制消息（`ncclIbPostFifo` 的游标从 `devIndex=0` 开始），从而控制流可以走与数据不同的 RoCE traffic class。注意：merged devs 为 2 时 QP 1 上的 FIFO post 不生效（已知局限）。
3. **告警改进**：完成错误 WARN 附加 HCA 名（`pd->context->device->name`）；TRACE 的 `wr_id` 格式修为 `%ld`；IB 连接 INFO 日志增加 subnet-prefix 与 FLID。
   （同 commit 的其他文件：`p2p_plugin.c` 混合单/多端口 NIC 时禁用 merge；`socket.c` abort 标志改 `__ATOMIC_ACQUIRE` 读取。）

### 6. 数据路径

**发送（ncclIbIsend，:1376）**：轮询 `fifo[fifoHead%64]` 等接收端 RDMA 写入的 CTS（`slots[0].idx == fifoHead+1` 后自旋等齐 `nreqs` 个，`__sync_synchronize()`），按 tag 匹配空闲 `fifoReqs` slot，size 截断到对端 posted 大小；凑齐全部 `nreqs`（最多 8）才调 `ncclIbMultiSend`，清 slot、`fifoHead++`。

**ncclIbMultiSend（:1272）**：构造 WR 链——每个 request 一个非 signaled `IBV_WR_RDMA_WRITE`，末尾一个 signaled `WRITE_WITH_IMM`。单发送 imm=实际 size；多发送最后把各 request size RDMA 写入对端 `remSizesFifo`。adaptive routing（`comm->ar && size > NCCL_IB_AR_THRESHOLD(8192)`）时，数据走纯 RDMA_WRITE，另发 0 字节 WRITE_WITH_IMM 触发对端完成。多 QP 分片：`NCCL_IB_SPLIT_DATA_ON_QPS` 开则按 `base.nqps` 分，否则按 `ndevs` 分，每片 128B 对齐；rkey 按 `qp->remDevIdx`、lkey 按本地 dev 选取。`wr_id` 每字节打包一个 request 索引（最多 8 个）。

**接收（ncclIbIrecv，:1540）**：分配 request → 在每个 QP 上 post 空 recv WR（承接对端 WRITE_WITH_IMM）→ `ncclIbPostFifo` 填充 `remFifo.elems[slot]`（addr/size/tag/idx/全部 dev 的 rkey）并 RDMA_WRITE 进发送端 FIFO；仅当 `slot == ctsQp->devIndex` 时才 signaled——保证每个发 FIFO 的 QP 周期性被排空而不必每 post 一个 signaled WR。

**Flush（ncclIbIflush，:1587）**：仅对最后一个非零 size 的 recv、且 `flushEnabled` 时：每 dev 在自连接 flush QP 上发 signaled 1 字节 `IBV_WR_RDMA_READ`，把对端 GPU 内存读入 host——强制之前的 RDMA write 对 GPU 可见（GDR flush）。

**完成（ncclIbTest，:1624）**：request 完成条件 `events[0]==events[1]==0`；否则每 dev CQ 一次 poll 最多 4 个 WC。错误路径 WARN 打印对端地址、wc status/opcode/byte_len/vendor_err、request 类型、RoCE 时的 GID 串、HCA 名（b246b19），返回 `ncclRemoteError`。`wr_id & 0xff` 定位 request，组内其余 request 逐字节递减 events。

**异步事件线程**（p2p_plugin.c 的 `ncclIbAsyncThreadMain`）：每设备一个 detached 线程，对除 `IBV_EVENT_COMM_EST` 外的事件打 WARN（此版本尚不计数致命错误）。

### 7. 调优参数（b246b19 版 ib_plugin.c 内的 NCCL_PARAM）

| 环境变量 | 默认值 | 备注 |
|---|---|---|
| `NCCL_IB_GID_INDEX` | -1（自动） | RoCE GID |
| `NCCL_IB_ROUTABLE_FLID_GID_INDEX` | 1 | **b246b19 新增**，IB 可路由 FLID GID |
| `NCCL_IB_ROCE_VERSION_NUM` | 2 | |
| `NCCL_IB_IS_GLOBAL` | 0 | |
| `NCCL_IB_TIMEOUT` | 18 | |
| `NCCL_IB_RETRY_CNT` | 7 | |
| `NCCL_IB_PKEY` | 0 | |
| `NCCL_IB_USE_INLINE` | 0 | |
| `NCCL_IB_SL` | 0 | |
| `NCCL_IB_TC` | 0 | |
| `NCCL_IB_AR_THRESHOLD` | 8192 | |
| `NCCL_IB_PCI_RELAXED_ORDERING` | 2（自动） | |
| `NCCL_IB_FIFO_TC` | 0（禁用） | **b246b19 新增** |
| `NCCL_IBEXT_DISABLE` | 0 | |
| `NCCL_IB_MERGE_VFS` / `NCCL_IB_MERGE_NICS` | 1 / 1 | |
| `NCCL_IB_QPS_PER_CONNECTION` | 1 | |
| `NCCL_GDR_FLUSH_DISABLE` | 0 | |
| `NCCL_IB_SPLIT_DATA_ON_QPS` | 0 | |

另：`NCCL_IB_ADDR_FAMILY`/`NCCL_IB_ADDR_RANGE`/`NCCL_IB_HCA` 直接读环境变量；`NCCL_IB_ADAPTIVE_ROUTING`（-2）、`NCCL_SHARP_MAX_COMMS`（1）在 p2p_plugin.c。

### 8. 其他值得注意的点

- `ncclIbLock` 全局锁声明了但全仓库未使用（死代码）；实际同步靠 `nccl_p2p_lock`（一次性 init）、per-dev `ncclIbDev.lock`（PD 引用计数与 MR cache）；数据路径无锁，靠 FIFO idx 轮询 + `__sync_synchronize`。
- MR 注册一次覆盖 merged devs 全部子设备；relaxed ordering 开启时用 `wrap_ibv_reg_mr_iova2` 并加 `IBV_ACCESS_RELAXED_ORDERING`。
- 若干小瑕疵：`ncclIbIsend` 的 `ready==0` 检查重复一次（不可达）；`ncclIbTest` 中 `sizes[0]=r->send.size` 赋值两次；`:874` 有 `ece_supported && ece_supported` 重复条件；`static pthread_t ncclIbAsyncThread` 只存一个线程句柄但每设备建一个线程（detached，无 join）。

---

## 第二部分：与 master 的对比

### 1. 总体概况

| 维度 | b246b19（2024-06） | master（2026-04） |
|---|---|---|
| `ib_plugin.c` 行数 | 1851 | 2335 |
| 插件 API | `ncclNetPlugin_v5…v8`（`IBext_v5…v8`） | `ncclNetPlugin_v6…v11`（v5 及 `net_v5.h` 已删除） |
| 每 NIC 最大子设备 | 2 | 4（`NCCL_IB_MAX_DEVS_PER_NIC`） |
| merged devs 上限 | `MAX_IB_DEVS` | `MAX_IB_VDEVS = MAX_IB_DEVS*8 = 256` |
| 致命异步错误处理 | 仅 WARN | 计数 + 在 isend/irecv/test 中上抛 `ncclSystemError` |
| 虚拟设备（vFFT / NIC fusion） | 无 | `makeVDevice` + `vProps` 全链路 |
| Direct NIC（mlx5 data-direct） | 无 | 有 |
| TC/SL 语义 | 静态参数（默认 0） | 默认 -1=未设置，可由 NCCL 经 `ncclNetCommConfig_t` 下发并在连接间协商 |
| isend/irecv size 类型 | `int` | `size_t`（v5–v8 包装函数做收窄） |

10 个改动 `ib_plugin.c` 的 commit 可分为：4 次上游 NCCL 同步（2.23.4 / 2.24 / 2.26 / 2.28.7-net-v11）、3 个 ECE 修复、Direct NIC 特性、版权头规范化、错误信息增强。master 上 `ib_plugin.c` 之后还有 SHARP overlap、collect nic-fusion、suse16 构建修复等 commit，但均不再触碰此文件。

### 2. 逐 commit 分析（按时间顺序）

**① `39fe29d` "Update with nccl-2.23.4-1"（2024-09）——异步致命错误传播 + 健壮性**

- 新增 `NCCL_IB_RETURN_ASYNC_EVENTS`(=1)、`NCCL_IB_ECE_ENABLE`(=1)；`NCCL_IB_TIMEOUT` 默认 18→20。
- 新增 `struct ncclIbStats { int fatalErrorCount; }`，嵌入 `ncclIbNetCommBase` 与 `ncclIbDev`；异步线程按事件类型调 `ncclIbDevFatalError/ncclIbCqFatalError/ncclIbQpFatalError`——通过 CQ/QP 的 context 指针（新增 `cq_context`/`qp_context` 参数传入 `&comm->base.stats`）定位到具体 comm 并计数。
- `ncclIbIsend`/`ncclIbIrecv`/`ncclIbTest` 入口及 CQ poll 循环中调用 `ncclIbStatsCheckFatalCount()`，发现致命事件即 WARN 并返回 `ncclSystemError`——**结束了"异步错误只能等完成超时"的局面**。
- `ncclIbSendComm` 字段重排（`fifo`/`sges`/`wrs` 提前，对齐要求）；ECE 查询受 `NCCL_IB_ECE_ENABLE` 门控。
- `ncclIbListen/Connect/Accept/RegMrDmaBuf` 错误处理重构为 `NCCLCHECKGOTO(..., fail)`，fail 路径关 socket、释放 comm（修泄漏）。
- 小修：`envIbAddrRange` 掩码解析 off-by-one；RoCE 版本 sysfs 读取失败告警；`ncclIbTest` 加 NULL 防护。

**② `2a632df` "Fix ece check"（2024-10）**

- `ncclIbConnect`/`ncclIbAccept` 中 `memset(&meta, 0, sizeof(meta))`——此前 `ncclIbConnectionMetadata`（含 `ece_supported`）**未初始化就发给对端**；accept 侧对端不支持 ECE 时显式置 0。

**③ `85b73d3` "update with nccl-2.24-1"（2024-10）——最大的一次变更：net v9 + 虚拟设备（vFFT/NIC fusion）**

- 新增 `include/net_v9.h`：`ncclNetProperties_v9_t` 增加 `vProps`（`ncclNetVDeviceProps_t{ndevs, devs[4]}`）、`maxP2pBytes`、`maxCollBytes`；`ncclNet_v9_t` 新增 **`makeVDevice`** 回调；`NCCL_NET_OPTIONAL_RECV_COMPLETION` 标志。
- 结构变化：`NCCL_IB_MAX_DEVS_PER_NIC` 2→4；`ncclIbMergedDev` 改为 vProps 表达，`dmaBufSupported` 移到 `ncclIbDev`（并新增 `virtualPciPath`、`latency`）；`ncclIbNetCommBase.ndevs` → `vProps`，新增 **`nDataQps`**；`ncclIbRequest.events[]/devBases[]` 扩为 4 槽；`ncclIbSendFifo.size` int→uint64_t。
- 新增 `ncclIbMakeVDevice()`：允许 NCCL 在运行时把任意物理 IB 设备融合成一个逻辑 NIC；物理设备本身也通过同一 helper 注册为 vDev。
- 连接建立新增 `SendDevList`/`RecvDevList` 两个状态：双方在 metadata **之前先交换 vProps**，`nqps = qpsPerConn * max(本地ndevs, 对端ndevs)`，使两端融合度不一致也能配对 QP；accept 侧新增 `ncclIbCheckVProps()` 交集检查与 `NCCL_IB_WARN_RAIL_LOCAL`(默认0) 参数。
- API 拓宽：isend size 改 `size_t`、irecv sizes 改 `size_t*`，为 v5–v8 增加收窄包装；新增 `ibPlugin_v9`。
- **TC/SL 语义变化**：`NCCL_IB_TC` 默认 0→-1；`ncclIbRtrQp` 的 `override_tc` 更名 `fifoTc`，逻辑变为 `fifoTc && IB_FIFO_TC != -1 ? IB_FIFO_TC : IB_TC`——**`IB_FIFO_TC=0` 现在是真实 TC 值，-1 才是"禁用"**（b246b19 中 0 表示禁用）；且 accept 侧**所有** QP 都传 `fifoTc=true`（b246b19 仅 QP 0，修复了 merged devs 下覆盖不全的局限）。
- MTU 协商移到 connect 侧（RTR 前取 MIN）；修复 adaptive routing 判错设备的 bug（`ncclIbDevs[dev].ar` → `ncclIbDevs[ibDevN].ar`）。
- `wrap_ibv_modify_qp` 重写：失败时打印本地/对端 GID。

**④ `389e247` 版权头 SPDX 规范化**：无功能变化。

**⑤ `98eb504` "nccl-2.26 update"（2025-03）——net v10 + NCCL 下发 traffic class**

- 新增 `include/net_v10.h`：`init` 增加 `ncclProfilerCallback_t`（IB 后端接收但未使用）；`connect` 增加 `ncclNetCommConfig_t* config`（携带 `trafficClass`）；isend/irecv 增加 `phandle`/`phandles`（设备卸载句柄，IB 路径未使用）。
- **SL/TC 交换**：`NCCL_IB_SL` 默认 0→-1，新增 `NCCL_IB_SL_DEFAULT`/`NCCL_IB_TC_DEFAULT`(0)；`ncclIbConnectionMetadata` 增加 `int tc; int sl`，按 `参数 != -1 ? 参数 : (config->trafficClass != UNDEF ? config : DEFAULT)` 计算，**responder 回显 requestor 的 sl/tc**；`ncclIbRtrQp` 改为接收 `tc, sl` 参数而非直接读全局参数——SL/TC 从"本端静态配置"变为"连接级协商值"。
- 链路层不匹配（本地混合 / 本地与对端不一致）直接 WARN 中止；RoCE 版本读取 `EINVAL` 视为成功（容器无 GID sysfs 场景）。

**⑥ `b2e92a4` "Fix for ECE options not being exchanged correctly"（2025-04，移植 NCCL MR!817）**

- responder 侧的 `wrap_ibv_query_ece`（读回协商后的 ECE）移到 **RTR/RTS 之后**，且仅当**双方**都支持 ECE 时才查询——此前在 RTR 前查、只看本地标志，回传给 requestor 的 ECE options 可能是错的/过期的。最终 accept 流程：`set_ece → RtrQp → RtsQp → (双方都支持才) query_ece 回传`。

**⑦ `2b7ac9a` "Direct NIC support"（2025-06）——mlx5 data-direct DMA**

- 含义：ConnectX 系列 mlx5 设备支持 "data direct" DMA（设备直接 DMA 到 GPU 内存，绕开传统 DMA-buf 路径），插件用 `mlx5dv_reg_dmabuf_mr()` + `MLX5DV_REG_DMABUF_ACCESS_DATA_DIRECT` 注册 MR，并把这类端口作为独立 NIC 暴露给 NCCL。
- 构建：`configure.ac` 新增 `--with-mlx5-dv`（默认自动探测），检查 libmlx5、`mlx5dv.h` 及 `mlx5dv_is_supported/get_data_direct_sysfs_path/reg_dmabuf_mr` 等符号，定义 `HAVE_MLX5_DV`。
- 发现（p2p_plugin.c）：每个 active 端口可枚举为两个 `ncclIbDev`——普通 `mlx5_X` 与 `mlx5_X_dma`（devName 加 `"_dma"` 后缀，pciPath 用 data-direct sysfs 路径，`capsProvider.mlx5.dataDirect=1`）；新参数 **`NCCL_IB_DATA_DIRECT`**（默认 1：只暴露 `_dma` 设备；2：两者都暴露；0：禁用）。data-direct 设备 `getProperties` 上报 `forceFlush=1`。
- `ib_plugin.c` 本体改动很小（~7 行）：`ncclIbRegMrDmaBufInternal` 按 `dataDirect` 标志分支到 `wrap_mlx5dv_reg_dmabuf_mr(..., mlx5_access=1)`。后续 commit 还补充了 CX8 的 200G/lane 速率识别、`active_speed_ex`、SHARP 侧 `_dma` 后缀剥离与 `SHARP_COLL_REG_FIELD_DMABUF_DATA_DIRECT` 传递等。

**⑧ `6d4b1d8` "net v11 plugin update"（2025-09）——按 communicator 的 init**

- 新增 `include/net_v11.h`：`init(void** ctx, uint64_t commId, ncclNetCommConfig_t* config, log, prof)`——NCCL 现在每个 communicator 调一次 init；`listen`/`connect` 接收 `ctx`；新增 **`finalize`** 与 **`setNetAttr`** 回调；properties 增加 `maxMultiRequestSize`。
- `ib_plugin.c` 用静态 `ibContext` + 引用计数维持"设备只初始化一次"，`config->trafficClass` 存入 ctx 供 connect 使用；`ncclIbSetNetAttr` 为 no-op 桩。
- 删除 `ibPlugin_v5` 与 `net_v5.h`，支持区间变为 v6–v11。

**⑨ `97b2d10` "Improve IB completion/async error WARN messages"（2025-10，移植 NCCL MR!1301）**

- 新增 `ibvWcStatusStr()`/`ibvWcOpcodeStr()`；完成错误 WARN 打印字符串化 status/opcode、`reqSize`（发送侧用 `req->send.size`，因 `wc.byte_len` 不可靠）、vendor_err、GID、HCA 名。b246b19 加入的 HCA 名告警在此进一步丰富。

**⑩ `ebb449d` "net v11 update (v2.28.7-1)"（2025-10）**

- `net_v11.h` 对齐 NCCL 2.28.7：新增 `ncclGin_v11_t`（GPU-Initiated Networking：iput/ginProgress/regMrSym 等，IB 后端未实现）；`net.h` 增加 `NCCL_NET_MR_FLAG_FORCE_SO`、signal 操作等。
- `ncclIbInit` 拆分为引用计数的 `ncclIbInitDevices/FinalizeDevices`；per-comm ctx 改为 malloc 的 `ncclNetCommConfig_t`（finalize 释放）。
- MR 注册内部函数增加 `mrFlags` 参数：始终带 `IBV_ACCESS_REMOTE_ATOMIC`，`NCCL_NET_MR_FLAG_FORCE_SO` 时抑制 relaxed ordering——为 GIN 预留的管线，导出接口仍传 0。

### 3. 主题深入对比

#### 3.1 b246b19 的 FLID 多子网路由在 master 中的存续

**完整保留、几乎未动**。`ncclIbExtractLocalSubnetPrefix`、`ncclIbExtractFlid`、`ncclIbGetGidIndex`（含 `IB_ROUTABLE_FLID_GID_INDEX` 探测）在 b246b19 与 master 之间**逐字节一致**；`ncclIbRtrQp` 的 `!same_subnet` 分支（FLID 作 dlid、FLID==0 时 WARN 回退 LID）也不变。`ncclIbRtrQp` 的全部差异仅在于：`override_tc`→`fifoTc` 改名、新增 `tc`/`sl` 参数、一行 TRACE。

#### 3.2 FIFO TC 的语义漂移（兼容性注意点）

| | b246b19 | master（2.24 起） |
|---|---|---|
| `NCCL_IB_FIFO_TC` 默认 | 0 = 禁用 | 0 = **真实 TC 值 0**；-1 = 禁用 |
| `NCCL_IB_TC` 默认 | 0 | -1（未设置则用 `NCCL_IB_TC_DEFAULT`=0 或 NCCL 下发值） |
| 生效 QP | 仅接收侧 QP 0 | 接收侧全部 QP |
| TC/SL 来源 | 仅本地环境变量 | 连接元数据中协商（requestor 计算，responder 回显） |

行为变化：在 master 上设 `NCCL_IB_FIFO_TC=0` 会**强制 FIFO 流量走 TC 0**，而 b246b19 上是"不覆盖"。跨版本混布时需注意。

#### 3.3 net v9/v10/v11 API 演进一览

| 方面 | b246b19（最高 v8） | master（最高 v11） |
|---|---|---|
| `init` | `(logFunction)` | `(void** ctx, commId, config, log, prof)`（v11 按 comm 调用） |
| `getProperties` | v8 属性集 | +`forceFlush`、`vProps`、`maxP2pBytes/maxCollBytes`(1TB)、`maxMultiRequestSize` |
| `connect` | `(dev, handle, sendComm)` | +`ctx`（内含 trafficClass） |
| `isend`/`irecv` | `int size(s)` | `size_t size(s)` + `phandle(s)`（IB 未用） |
| `makeVDevice` | 无 | 已实现（NIC fusion） |
| `finalize`/`setNetAttr` | 无 | finalize 释放 per-comm ctx；setNetAttr 为桩 |
| `getDeviceMr`/`irecvConsumed` | NULL | 仍为 NULL（无设备卸载） |

#### 3.4 未变的机制

sizesFifo/remSizesFifo 多请求回传、per-connection QP 数组与 `NCCL_IB_SPLIT_DATA_ON_QPS`、adaptive routing（0 字节 WRITE_WITH_IMM 尾部触发）、GDR gpuFlush 自连接 QP、ECE 协商框架、MR cache、merged devs 融合逻辑——这些 b246b19 的核心机制在 master 上全部保留，仅做了 vProps 化、拓宽与修复。

#### 3.5 参数差异汇总

- **新增**：`NCCL_IB_RETURN_ASYNC_EVENTS`(1)、`NCCL_IB_ECE_ENABLE`(1)、`NCCL_IB_WARN_RAIL_LOCAL`(0)、`NCCL_IB_DATA_DIRECT`(1，在 p2p_plugin.c)、`NCCL_IB_SL_DEFAULT`/`NCCL_IB_TC_DEFAULT`(0)。
- **默认值变化**：`NCCL_IB_TIMEOUT` 18→20；`NCCL_IB_TC` 0→-1；`NCCL_IB_SL` 0→-1；`NCCL_IB_FIFO_TC` 数值不变(0)但语义反转（0=生效，-1=禁用）。
- **保持不变**：`IB_GID_INDEX`(-1)、`IB_ROUTABLE_FLID_GID_INDEX`(1)、`IB_ROCE_VERSION_NUM`(2)、`IB_IS_GLOBAL`(0)、`IB_RETRY_CNT`(7)、`IB_PKEY`(0)、`IB_USE_INLINE`(0)、`IB_AR_THRESHOLD`(8192)、`IB_PCI_RELAXED_ORDERING`(2)、`IB_QPS_PER_CONNECTION`(1)、`IB_SPLIT_DATA_ON_QPS`(0)、`IB_ADAPTIVE_ROUTING`(-2)、`IB_MERGE_VFS/NICS`(1)、`IBEXT_DISABLE`(0)、`GDR_FLUSH_DISABLE`(0)。

---

## 总结

1. **b246b19 的定位**：这是 2024 年 6 月基于 net v5–v8 API 的版本，该 commit 自身贡献了三个生产级特性——IB 跨子网 FLID 路由（`IB_ROUTABLE_FLID_GID_INDEX`）、RoCE FIFO 控制流独立 TC（`IB_FIFO_TC`）、错误告警增强（HCA 名），外加混合 NIC 禁用合并与 socket 原子读两个修复。其架构（OOB socket 三段握手 + 可恢复状态机、FIFO/CTS 信用机制、多 QP 条带化、GDR flush QP、MR cache）与当时上游 NCCL 内部插件一致。
2. **master 的演进主线**：四次上游同步把 API 从 v8 推到 v11，带来四类能力——(a) **虚拟设备/NIC fusion**（v9，每 NIC 4 子设备、运行时融合、vProps 协商）；(b) **按 communicator 的上下文与 NCCL 下发 TC/SL**（v10/v11，SL/TC 成为连接级协商值）；(c) **可靠性**（异步致命错误计数上抛、ECE 三处修复、错误信息字符串化、fail 路径资源清理）；(d) **Direct NIC**（mlx5 data-direct DMA-buf，`_dma` 设备枚举）。
3. **延续性**：b246b19 的 FLID 路由代码在 master 中原样保留；sizesFifo、AR、gpuFlush 等核心数据路径机制均未重构。两个版本的主要**行为差异**集中在 TC/SL 默认值语义（0→-1 体系）与 FIFO TC 生效范围上，这是从 b246b19 升级或混布时最需要关注的兼容点。
