# NCCL GIN（GPU-Initiated Networking）功能与实现详解

## 一句话总结

GIN（GPU-Initiated Networking）是 NCCL 内部的 GPU 发起网络抽象层：**collective kernel 在 GPU 侧直接构造并提交网络操作**（Put/Get/Flush/Signal），绕过传统"GPU 提交工作 → CPU proxy 线程 → 网卡"的 host 转发路径，从而降低跨节点通信延迟。GIN 没有公开用户 API，使用者是 contrib 插件（`nccl_ep`/`nccl_m2n`/`nccl_ubx`/`nccl_checkpoint`）、Device API（如 alltoall-GIN）以及 symmetric memory / RMA 路径。

---

## 1. 背景与动机

传统 NCCL 网络路径中，GPU kernel 通过 P2P 消息把工作描述写给 CPU 侧 proxy，由 CPU proxy 调用网络插件（IB/socket）发起传输。每次跨节点操作都有 host 参与，带来延迟与 CPU 开销。

GIN 的目标：

- **GPU 直接发起**：kernel 内写描述符（GFD, GPU Function Descriptor）到队列或直接写网卡，CPU 不再处于关键路径。
- **统一的 device 抽象**：一套 device API（`ncclGinApi_*`）屏蔽不同后端，编译期/运行期按 `backendMask` 分发。
- **可插拔**：host 侧通过 GIN plugin vtable（`ncclGin_v14_t`）接入，支持外部厂商实现 GPUDirect-Async 方案。

---

## 2. 总体架构

GIN 由两层组成：

```
                    NCCL collective kernel / Device API
                                   │
                    ┌──────────────▼───────────────┐
                    │  ncclGinApi_* (device 侧)      │
                    │  Put / PutValue / Get / Wait   │
                    │  Flush / FlushAsync / Reset... │
                    └──────────────┬───────────────┘
                                   │ backendMask 分发
              ┌────────────────────┼────────────────────┐
              │                    │                    │
        ┌─────▼──────┐       ┌─────▼──────┐       ┌─────▼──────┐
        │ GIN Proxy  │       │  GDAKI     │       │    GPI     │
        │ GPU→GFD队列 │       │ GPU 直接写  │       │ GPU 直接写  │
        │ CPU proxy  │       │ DOCA QP    │       │ NIC 硬件队列│
        │ →RMA plugin│       │ (GPUNetIO) │       │ (外部 plugin)│
        └─────┬──────┘       └────────────┘       └────────────┘
              │
        RMA plugin (ncclRma_v14_t, 如 ncclRmaIbProxy)
```

三个后端的本质区别是 **CPU 参与数据面的程度**：

| | GIN Proxy | GDAKI | GPI |
|---|---|---|---|
| 数据面 CPU | 有（proxy 线程解析 GFD 并调 RMA plugin） | 无（GPU 直接提交到 DOCA QP） | 无（GPU 写硬件队列，NIC 直接消费） |
| 提交描述符 | 128B GFD（16 × 8B qword） | DOCA device verbs API 调用 | 64B GFD（8 × 8B segment） |
| NCCL 内置 host plugin | 是（`ncclGinProxy`） | 是（`ncclGinIbGdaki`） | 否（外部 NIC 厂商 `libnccl-gin.so`） |
| 所需硬件 | 无特殊要求（需 GDR/DMA-BUF 加速） | MLX5 NIC + DOCA GPUNetIO | 支持 GPI 的下一代 NIC |
| 版本门槛 | NCCL 2.30.3+ | NCCL 2.30.3+ | NCCL 2.30.5+ |

GIN 类型定义在 `src/include/nccl_device/core.h`：`NCCL_GIN_TYPE_NONE=0 / PROXY=2 / GDAKI=3 / GPI=4`，其值必须与 `src/include/nccl_device/net_device.h` 中的 `NCCL_NET_DEVICE_GIN_*` 一致。

---

## 3. 使能条件与初始化流程

GIN 的启用分三层把关：

### 3.1 编译期

`src/include/nccl_device/gin/gin_device_common.h`：

- `NCCL_GIN_PROXY_ENABLE`：默认 1。
- `NCCL_GIN_GPI_ENABLE`：`CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700`。
- `NCCL_GIN_GDAKI_ENABLE`：同上（sm_70+/CUDA 12.2+）。
- `NCCL_GIN_HAS_FENCE_ACQUIRE_RELEASE_PTX`：CUDA 12.8+ 且 sm_90+。

`src/plugin/gin.cc` 的 `ncclGinInit` 还要求 GPU 计算能力 ≥ 70（Volta 起）。

### 3.2 运行期插件探测（`ncclGinInit`）

插件加载顺序（`initPluginLibsOnceFunc`）：

1. `NCCL_GIN_PLUGIN` 指定的外部插件（默认 `libnccl-gin.so`，可逗号分隔多个）。
2. NET 插件（`libnccl-net.so`）里暴露的 GIN vtable（`getNcclGin`）。
3. 内置 GDAKI：`ncclGinIbGdaki`（仅 MLX5 NIC，且需 GDR）。
4. 内置 GIN Proxy 兜底：`ncclGinProxy`（包装 RMA plugin，`ncclGinProxyInit` 取 `comm->rmaState.rmaProxyState.ncclRma`）。

按版本 v14 → v13 依次 `getNcclGin`；`NCCL_GIN_TYPE` 可强制指定后端（不匹配的插件被跳过）。外部 plugin 若声明 `NCCL_NET_DEVICE_GIN_PROXY` 会被跳过（避免与外置 RMA backend 冲突）。初始化成功后：

```c
comm->sharedRes->ginState.ncclGin = gin;
comm->sharedRes->ginState.ginType = (ncclGinType_t)props.netDeviceType;
comm->sharedRes->ginState.supportsStrongSignals = ginProperties.supportsStrongSignals;
comm->sharedRes->ginState.supportsVASignals = ginProperties.supportsVASignals;
```

### 3.3 通信子（devComm）激活

`src/dev_runtime.cc`（约 1110-1173）：

- 必须显式请求 `ginConnectionType = NCCL_GIN_CONNECTION_FULL / RAIL`（旧 `ginForceEnable` 已废弃，等价 FULL）。
- 所有 rank 的 `supportedGinType` 必须一致（`src/init.cc` 计算 `globalGinSupport`）；跨 NIC 能力决定 FULL 还是 RAIL。
- 需要 `cuMemGdrSupport` 且没有 MloPart GPU（`init.cc:1668-1670`）。
- 满足后置 `devr->ginEnabled = true`，并把 `ginConnectionsRailed` 写入 devComm。

**普通 `ncclCommInitRank` 默认 `NCCL_GIN_CONNECTION_NONE`，因此标准 PyTorch 训练不会启用 GIN。**

---

## 4. 核心概念与抽象

### 4.1 connection 与 context

- **GIN connection**：一个 (src, dst) 网络设备对，由 `listen/connect` 建立，host 侧 opaque 对象是 `collComm`。数量很少（`NCCL_GIN_MAX_CONNECTIONS=4`），通常一个 rank 对每个本地 NIC 一条。
- **GIN context**：connection 下的子资源（一个 context ≈ 一个 QP / 一个队列通道），用于提升并行度。每个 context 有固定的 counters/signals/队列，**ordering 保证以 context 为界**（同一 context 内 signal 完成意味着之前所有 put 完成）。

`ncclGinDevCommSetup`（`src/gin/gin_host.cc`）按 `reqs->ginContextCount` 分配 context，向上取整到 `ginCommCount` 的倍数；v13 以下插件只支持每 connection 一个 context。`createContext` 返回 host 侧 `ginCtx` 和 device 侧 `ncclNetDeviceHandle_t`（`handle` 是 GPU 可见的 context 数组）。

### 4.2 内存注册（对称窗口）

`ncclGinRegister`（`gin_host.cc`）对每个 connection 调 `regMrSym`，得到 host 侧 `mHandle` 与 device 侧 `ginHandle`（即 `ncclGinWindow_t`）。注册是**对称的**：调用发生在所有 rank 上，返回句柄足以寻址本地与所有 peer 的缓冲区。

- 多段（multi-segment）注册要求所有 connection 支持 `NCCL_PTR_DMABUF`。
- `NCCL_WIN_STRICT_ORDERING` → `NCCL_NET_MR_FLAG_FORCE_SO`（强制 strict ordering）。
- GIN Proxy 的注册会优先尝试 DMA-BUF（`cuMemGetHandleForAddressRange` + `regMrSymDmaBuf`），失败回退普通注册。

### 4.3 操作语义

| 操作 | 语义 |
|---|---|
| Put | 本地源 → 远端目标；不保证请求间顺序 |
| PutValue | 小数据内联进描述符的 Put（Proxy 最多 12B；GPI 内联 8B） |
| Get | 远端源 → 本地目标；不保证顺序，数据可见性要靠 Flush |
| Signal | 给目标地址加固定值；强信号保证此前所有 put/signal 可见，弱信号只保证捆绑的 put |
| Flush | 本地完成屏障：put 表示源可复用，get 表示数据可见可用 |
| Counter | 本地 64 位计数（每完成一个捆绑操作 +1）；只保证本地完成 |

Signal 分两种：**Indexed signal**（`createContext` 时由 plugin 分配，通过 `signalId` 索引，`GetSignalPtr` 返回可原子读的地址）和 **VA signal**（用户注册的任意 window/offset，由 device 代码直接写）。

### 4.4 其它约定

- `coop`：协作线程组参数，用于确定临界区由单线程执行、并行提交等。
- `abortFlag`：阻塞等待（Wait/Flush）时轮询的中止标志，非空则提前返回。
- `DescriptorSmem`：64B 共享内存 scratch，可选地复用来避免每次调用重建内部结构（GIN Proxy 直接把 GFD 放这里）。
- `ncclGinOptFlags`：`MaySkipCreditCheck`（跳过队列信用检查）、`AggregateRequests`（期望后续还有请求，可推迟批量收尾逻辑）。各后端按能力使用：GPI 据此跳过信用检查，GDAKI 映射为 DOCA 的 `SKIP_AVAILABILITY_CHECK` / `SKIP_DB_RINGING`，Proxy 目前不使用（始终做信用检查）。
- **backend version**：host/device 共享结构（context、window 等）格式变更时递增；`gin_host.cc` 保存映射表 `{proxy, gdaki}: {0, 2.30.3, 2.30.5}`、`gpi: {0, 2.30.5}`，`NCCL_DEV_API_JIT=1` 时固定取最新版本。

---

## 5. 后端一：GIN Proxy

GIN Proxy 是**默认兜底**且最完整的实现：GPU 写 GFD 到 ring queue，CPU progress 线程解析后转调 RMA plugin 的 `iput/iputSignal/iget/iflush`。

### 5.1 数据流

```
GPU kernel:
  1. buildGfd(): 构造 128B GFD（op/size/src,dst handle+offset/signal/counter/inline）
  2. postGfd():  原子 PI.fetch_add(1) 占 slot
                 (thread_rank==0) __stwt 16B×8 写入队列（st.global.wt, write-through）
                 队列满则等 CI（credit check）
  3. waitForGfdComplete(): 轮询 CI >= nextGfdIdx（rolling 比较防溢出）

CPU 线程 (NCCL GIN Progress%2d):
  ncclGinProgress → ginProgress(每 context):
    1. proxyGinPollCompletions(): rmaBackend->test() 查完成 → 更新 counter、推进 CI
    2. proxyGinPollGfd(): 看 GFD 首 qword flag → 拷贝 16 qword → 清零队列项 → SI++
    3. proxyGinProcessGfd(): 解析 op → iput/iputSignal/iget/iflush 提交到 RMA plugin
```

队列是 per-(context, peer) 的 lock-free multi-producer ring buffer；GPU 侧维护 PI（produce index），CPU 侧维护 SI（seen index）与 CI（consume index）。CI 在 GPU 可读内存中（GDR 允许时在 GPU 显存，否则 pinned host memory），GPU 忙等 `CI >= nextGfdIdx`。

### 5.2 GFD 格式（`gin_proxy_device_host_common.h`）

`ncclGinProxyGfd_t` 固定 **128B = 16 × 8B qword**，每个 qword bit0 是 valid flag（GPU 写、CPU 验证后清零）。关键 qword 布局：

```
[0] header        version(4b), size(57b)
[1]/[2]           inline 值 或 srcOff/srcHandle 或 VA signal off/handle（复用）
[3]/[4]           dstOff / dstHandle
[5] completion    counterId(23b), signalId(24b), signalValLow(16b)
[6] signalVal     isStrongSignal(1b), signalValLow2(16b), signalValHigh(32b)
[7] headerExt     op(16b)
```

op 位（`ncclGinProxyOp_t`）：`Put=1<<0`、`WithInline=1<<1`、`WithCounter=1<<2`、`WithSignalInc=1<<3`、`WithSignalAdd=1<<4`、`VASignal=1<<5`（纯 VA signal，无 put）、`Get=1<<6`、`Flush=1<<7`。

### 5.3 关键实现细节

- **大块拆分**：`DataChunkSize = 1GB`，Put/Get 超过则拆成多个 GFD。
- **PutValue/inline**：≤12B 数据直接放进 GFD；proxy 把 inline 值写到 host 侧 `inlines` 缓冲区（也注册了 MR），再以该 buffer 为源执行 `iput`。
- **带 signal 的 Put**：indexed signal 时 op 加 `WithSignalInc/Add`，proxy 用 `iputSignal`，signal 地址 = `(signalId + contextId*nSignalsPerContext) * 8`（`signals` 缓冲区以 `FORCE_SO` 注册，保证 put 与 signal 的次序）；**VA signal 必须拆成两个 GFD**（先不带 signal 的 put，再单独 VASignal GFD）。
- **Get 的可见性**：`lastIssuedGet`/`lastVisibleGet` 两个 per-peer 计数器；`FlushAsync` 记录 `lastIssuedGet` 与当前 PI；`Wait` 先等 CI，若 `lastVisibleGet < lastIssuedGet`，向本地队列投一个 `Flush` GFD（`iflush`）等它完成后更新 `lastVisibleGet`——保证 get 数据对 GPU 可见。
- **Counter 更新**：proxy 用 volatile 原子 load/store 对 GPU 可见 counter +1（GFD 带 `WithCounter` 时）。GPU 侧禁止在有 outstanding 操作时 reset counter。
- **Reset 不归零**：`signalOffsetsDev` 记录 reset 时的快照偏移，`GetSignalPtr` 返回 `ptr + offset`，省掉回写（backendVersion ≥ 2，同时给 signals MR 加 `NCCL_NET_MR_FLAG_SIGNAL_NEVER_RESET`）。
- **资源分配**：counters/queues/CI 通过 `allocMemCPUAccessible`（GDR 可用则放 GPU 内存，否则 pinned host）；PIs、signal offsets 在 GPU。
- **progress 线程**：`ncclGinProgress` 持有 comm 的 CPU affinity（NUMA-local），对 `devComms` 链表上每个 context 调 `ginProgress`；错误通过 `asyncResult` 上报，状态机 `1=运行 / -1=退出 / 0=等待 cond / -2=错误`。
- **队列深度**：默认 `NCCL_NET_MAX_REQUESTS * props.maxRecvs`，必须是 2 的幂，上限同默认值；`NCCL_GIN_PROXY_QUEUE_SIZE` 可覆盖。

---

## 6. 后端二：GDAKI（DOCA GPUNetIO）

GDAKI 让 GPU 通过 **DOCA GPUNetIO device verbs** 直接提交 IB 工作请求，数据面完全绕开 CPU（仅初始化/连接由 host 完成）。

### 6.1 实现位置

- Host：`src/transport/net_ib/gin.cc`（vtable `ncclGinIbGdaki`）+ `src/transport/net_ib/gdaki/gin_host_gdaki.cc`（QP 建立、AH、MR、context）。
- Device：`src/include/nccl_device/gin/gdaki/gin_gdaki.h` + `gin_gdaki_device_host_common.h`。

### 6.2 device 上下文

```c
struct ncclGinGdakiGPUContext {
  doca_gpu_dev_verbs_qp* gdqp;           // 每 peer 一个 QP
  doca_gpu_dev_verbs_qp* companion_gdqp; // 与 counter 操作配对的"伴生 QP"
  ncclGinGdakiGlobalGPUBufferTable<uint64_t> counters_table, signals_table; // rkeys/lkey/offset
  __be32 sink_buffer_lkey;               // 本地 sink buffer（signal/counter 的写目标）
  uint64_t* last_issued_get, *last_visible_get;
};
```

`gdakiCreateVerbsAh` 建立 address handle，`gdakiConnectQp` 把 GPU QP 与远端连接。

### 6.3 操作实现

- **Put**：`doca_gpu_dev_verbs_put[_signal[_counter]]`，参数：raddr（远端 addr+rkey）、laddr（本地 addr+lkey）、大小；带 counter 时用 companion QP 发出计数写，带 signal 时 `SIGNAL_OP_ADD` 到远端 signal 地址。
- **PutValue**：`doca_gpu_dev_verbs_p / p_signal<T>` 内联数据。
- **Get**：`doca_gpu_dev_verbs_get` 返回 ticket，记录到 `last_issued_get[peer]`（+1，0 表示无 get）。
- **FlushAsync**：若有未完成 get，发 `doca_gpu_dev_verbs_mcst`（sink 写 + ticket）并记录 `sq_rsvd_index`；否则记录当前 `sq_rsvd_index`。
- **Wait**：`doca_gpu_dev_verbs_wait(qp, pollIdx)` 轮询 CQ 到指定 SQ 槽位；DOCA 已保证 acquire 语义，忽略 `ord`。
- **ResetCounter/ResetSignal**：直接清零 `counters_table.buffer` / `signals_table.buffer`（或用户 VA）。

### 6.4 约束

- 仅 MLX5（`ibProvider == IB_PROVIDER_MLX5`），且需要 DOCA GPUNetIO 库。
- GPU 与 NIC 间需要 GDR；按注释 GDAKI 不支持 nv_peer_mem，依赖 DMA-BUF（`ncclGinIbGdrGpuSupport` 在 connect 时按 `CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED` 校验）。
- Traffic class：`NCCL_GIN_IB_TC`，缺省回退 `NCCL_IB_TC`。
- 不支持 `rankStride` 优化（`ncclGinIbGdakiCreateContext` 注释）。

---

## 7. 后端三：GPI（GPU Packet Interface）

GPI 是**最纯粹的 GPU 驱动方案**：GPU kernel 把 64B GFD 直接写入 NIC 硬件队列，NIC 自行解析执行；NCCL 内部**没有** GPI 的 host plugin，host 侧连接设置由外部 NIC 厂商插件（`NCCL_GIN_PLUGIN`）提供。

### 7.1 GFD 格式（`gin_gpi_device_host_common.h`）

`gpi_gfd_t` = **64B = 8 个 segment × 8B**，每个 segment bit0 为 owner 标签（= `pi >> log_depth`，防止队列复用时读到旧数据）：

```
[0] header       owner, op(8b), op_flags(8b), counter(16b), signal(16b)
[1] dst          owner, pe(31b, rank), size(32b)
[2] dst_handle   owner, dst_handle(16b), signal_value_low(32b)
[3] src_handle   owner, src_handle(16b), signal_value_high(32b)
[4] dst offset   owner, offset(63b)
[5] src offset   owner, offset(63b)
[6]/[7]          inline_data low/high
```

### 7.2 操作集

数据 op（`gpi_gfd_data_op`）：`READ=0`、`WRITE=1`、`WRITE_SIGNAL_ADD=2`、`WRITE_SIGNAL_COUNTED=3`、`WRITE_INLINE=4`、`WRITE_INLINE_SIGNAL_ADD=5`、`WRITE_INLINE_SIGNAL_COUNTED=6`、`AMO_ADD=7`（VA signal 用）、`PE_FLUSH=8`。

控制 op（`op |= GPI_GFD_OP_CTRL (1<<7)`）：`COUNTER_RESET=0`、`COUNTER_RESET_NO_WRITEBACK=1`、`COUNTER_WRITEBACK=2`、`SIGNAL_RESET=3`。

op flags：`WITH_COUNTER_FLAG(1<<0)`、`WITH_COUNTER_COUNTED(1<<1)`（每完成 +1）、`WITH_COUNTER_WRITEBACK(1<<2)`（完成后写回值）。

### 7.3 队列与提交

```c
typedef struct {
  uintptr_t* gpu_memic_ptr;  // ring buffer（NIC 可见）
  uint64_t pi_;              // produce index（GPU 原子加）
  gpi_ci_t* ci_;             // consume index（NIC 写）
  uint64_t ci_value_;        // CI shadow，减少读取
  uint32_t log_depth;        // 队列深度 = 2^log_depth
} Queue_t;
```

提交（`gpi_gpu_channel_post_gfd`）：

- **Thread 模式**：单线程用 128 位 MMIO store（`st.relaxed.sys.global.b128`，CUDA 12.3+；更老用 `v2.b64`）每次写两个 segment。
- **TMA 模式**（sm_90+，`GPI_USE_TMA_`）：`cp.async.bulk.global.shared::cta` 从 shared memory 拷贝 GFD，需 `DescriptorSmem`。

资源分享模式决定原子开销：`EXCLUSIVE`（普通 load/store）/ `CTA`（`thread_scope_block` 原子 + `ld/st.relaxed.cta`）/ `GPU`（`thread_scope_device`，默认）。

### 7.4 counter / signal / flush

- `gpi_counter_t`：64B 对齐 64 位；低 56 位成功计数，bit62 = error，bit63 = writeback pending。
- `gpi_signal_t`：128 位（value + flags），bit0 = `GPI_SIGNAL_COUNTED_FLAG`（counted 信号需要 `SIGNAL_RESET` 控制 op 清零）。
- **Flush**：每 peer 一个 `flush_tickets`（分配在 channel 之后、signals 之前）；`flushImplMode` 原子取 ticket → 发 `PE_FLUSH` GFD（counter = flush counter idx，`WITH_COUNTER_COUNTED|WRITEBACK`）→ GPU 忙等 `counter > ticket`。

### 7.5 版本要求

- 编译期 sm_70+/CUDA 12.2+；TMA 需 sm_90+。
- `gpiBackendMinVersions = {0, NCCL_VERSION(2,30,5)}` → **最低 NCCL 2.30.5**。

---

## 8. 后端分发机制

`src/include/nccl_device/gin/gin_device_common.h`：

```c
#define NCCL_GIN_BACKEND_MASK_ALL \
  (PROXY_ENABLE<<NCCL_NET_DEVICE_GIN_PROXY | GDAKI_ENABLE<<NCCL_NET_DEVICE_GIN_GDAKI | GPI_ENABLE<<NCCL_NET_DEVICE_GIN_GPI)
```

每个 API（`ncclGinApi_Put<backend>` 等）按后端模板特化；`ncclGinCallImpl(beMask, ctx, ...)` 运行期 switch 分发：

- 若 `beMask` 只有 1 个 bit（singleton），用 `__popc(beMask - 1)` 直接算出 case 编号，编译器可死代码消除其它后端。
- `ncclGinCtx_M<beMask>` 提供编译期已知 mask 的版本；`ncclGinCtx` 携带运行期 `backendMask`。
- 每个 case 内再校验 bit 存在，否则 `__builtin_unreachable()`。

`ncclGinCtx` 还携带 `rank/nRanks/handle/contextId/resourceSharingMode`，device 侧据此定位自己的 context 与队列。

---

## 9. 与 NCCL 其余部分的集成

### 9.1 device session 封装（`src/include/nccl_device/impl/gin__funcs.h`）

collective/Device API 代码通过 `ncclGin` 会话对象调用：

```c
ncclGinInitCommon(gin, comm, contextIndex):
  gin->connectionId = contextIndex % comm.ginConnectionCount;   // 4 条连接时的 modulo/位运算优化
  gin->contextId    = contextIndex / comm.ginConnectionCount;
  gin->_ginBackend  = comm.ginNetDeviceTypes[gin->connectionId];
  gin->_ginHandle   = comm.ginHandles[gin->connectionId];
```

- 窗口映射：`window->ginOffset4K * 4096 + offset` → GIN 偏移；`window->ginWins[connectionId]` → GIN window 句柄。
- 多段窗口：`ginMultiSegmentWins`，按 `findSegmentFromWindow` 分段后用 `getSegmentChunkSize` 逐段提交。
- Railed 连接：`teamRankToGinRank` 用 `idivFast32(worldRank, lsaSize, ...)` 把 world rank 映射为 GIN peer（RAIL 连接只覆盖同一 rail）。

### 9.2 谁在用 GIN

- `contrib/nccl_ep`（扩展插件，in/outbox、kernel fusion）、`nccl_m2n`、`nccl_ubx`、`nccl_checkpoint`。
- Device API 示例：`docs/examples/06_device_api/02_alltoall_gin`、`docs/examples/07_kernel_fusion/03_rmsnorm_gin`。
- RMA/symmetric memory 路径（参见 `rma-vs-gin.md` 与 `docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/`）。

### 9.3 与 RMA 的关系

GIN Proxy **在底层调用 RMA plugin**（`ncclRmaIbProxy` 的 `iput/iputSignal/iget/iflush`），二者共享 `src/transport/net_ib/gin.cc` 的 IB 实现：该文件同时导出 `ncclRmaIbProxy`（RMA plugin vtable）与 `ncclGinIbGdaki`（GIN plugin vtable）。区别只在"描述符谁构造"：RMA 由 host 构造，GIN 由 GPU kernel 构造 GFD。

---

## 10. 环境变量汇总

| 变量 | 默认 | 作用 |
|---|---|---|
| `NCCL_GIN_ENABLE` | 1 | 总开关 |
| `NCCL_GIN_TYPE` | -1（自动） | 强制 PROXY/GDAKI/GPI |
| `NCCL_GIN_PLUGIN` | `libnccl-gin.so` | 外部 GIN 插件列表（逗号分隔，`none` 禁用） |
| `NCCL_GIN_PLUGIN_REF_COUNT` | 0 | 插件引用计数 |
| `NCCL_GIN_NCONNECTIONS` | -2（自动） | GIN connection 数（≤4，所有 rank 取 min） |
| `NCCL_GIN_PROXY_QUEUE_SIZE` | -1（自动） | Proxy GFD 队列深度（须为 2 的幂） |
| `NCCL_GIN_IB_TC` | -1 | GIN IB traffic class（回退 `NCCL_IB_TC`） |
| `NCCL_DEV_API_JIT` | 0 | 1 时 device 代码固定用最新 backend version |

---

## 11. 源码地图

| 文件 | 作用 |
|---|---|
| `src/plugin/gin.cc` | 插件加载、版本协商、GIN 类型选择、`ncclGinInit/Finalize` |
| `src/gin/gin_host.cc` | host 状态机：connect once、devComm context 分配、progress 线程、MR 注册 |
| `src/gin/gin_host_proxy.cc` | Proxy host：GFD 轮询、RMA plugin 调用、counter/signal/CI 维护 |
| `src/gin/proxy_gpucontext/` | Proxy GPU context 初始化（v1/v2 兼容） |
| `src/include/gin/gin_host.h` | `ncclGinState`/`ncclGinStateDevComm` 定义 |
| `src/include/plugin/gin/gin_v13.h` / `gin_v14.h` | GIN host plugin vtable |
| `src/include/nccl_device/gin/gin_device_common.h` | device API 模板 + backendMask 分发 |
| `src/include/nccl_device/gin/proxy/` | Proxy GFD 格式与 device 侧 post/wait |
| `src/include/nccl_device/gin/gpi/` | GPI GFD 格式与 device 侧实现（64B GFD、MMIO/TMA） |
| `src/include/nccl_device/gin/gdaki/` | GDAKI device 侧实现（DOCA device verbs） |
| `src/transport/net_ib/gin.cc` | 内置 `ncclGinIbGdaki` 与 `ncclRmaIbProxy` vtable |
| `src/transport/net_ib/gdaki/gin_host_gdaki.cc` | GDAKI host：QP/AH/MR/context |
| `plugins/gin/` | 自定义 GIN 实现开发者指南 + 示例插件（`example/plugin.c`） |

---

## 12. 参考

- `plugins/gin/README.md`（自定义 GIN 实现指南：概念、host/device API、版本兼容）。
- `docs/contrib/GIN/NCCL_Gin_and_Symmetric_Memory/NCCL_Gin_and_Symmetric_Memory.md`。
- `rma-vs-gin.md`（RMA 与 GIN 对比、容易混淆的点）。
- `docs/dev_guide/nccl_internals.md` §10（GIN 三层使能条件总览）。
