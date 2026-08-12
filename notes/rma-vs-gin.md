# RMA vs GIN: NCCL 的内存访问与网络发起机制

## 一句话

- **RMA** 是用户可以调用的单边通信 API（`ncclPutSignal` / `ncclSignal` / `ncclWaitSignal`）。
- **GIN** 是 collective kernel 内部使用的 GPU 端网络基础设施，**没有**公开 API。
- **GIN Proxy 后端在底层调用的就是 RMA plugin**，但发起方是 GPU kernel 而非 host。

---

## 对比总览

| | RMA | GIN |
|---|---|---|
| **全称** | Remote Memory Access | GPU-Initiated Networking |
| **用户 API** | 是：`ncclPutSignal`, `ncclSignal`, `ncclWaitSignal` | **否**，完全内部使用 |
| **调用入口** | Host 端 CUDA stream 上 enqueue | GPU kernel 内 device 函数 |
| **数据流发起方** | host (`ncclLaunchRma`) 或 GPU (CE 路径不经过 host) | GPU kernel（写 GFD 到队列） |
| **不同节点** | RMA Proxy → RMA plugin (`iput`/`iputSignal`) | GIN Proxy → RMA plugin，或 GDAKI/GPI 直接写网卡 |
| **同节点** | CE 路径：CUDA copy engine D2D，不经过网络 | 通常不走 GIN |
| **后端/plugin** | `ncclRmaIbProxy` 等 (vtable `ncclRma_v14_t`) | 三种：GIN Proxy、GIN GDAKI、GIN GPI |
| **声明位置** | `src/nccl.h.in:652` (public API) | `src/include/nccl_device/gin/` (device headers) |
| **实现位置** | `src/rma/` | `src/gin/` (host) + `src/include/nccl_device/gin/` (device) |
| **同步模型** | stream 同步（`cuStreamBatchMemOp` wait-value） | counter/signal + GFD completion poll |
| **内存模型** | 对称窗口 (`ncclDevrWindow` via `ncclCommWindowRegister`) | GIN window (`ncclGinWindow_t`, opaque handle) |

---

## RMA 详解

### 用户视图

```c
// 1. 注册对称窗口（所有 rank 看到同一段内存）
ncclCommWindowRegister(comm, buff, size, &win, NCCL_WIN_COLL_SYMMETRIC);

// 2. 向 peer 写数据 + 发信号
ncclPutSignal(localBuff, count, ncclFloat32, peer, peerWin, offset,
              sigIdx, ctx, 0, comm, stream);

// 3. 只发信号
ncclSignal(peer, sigIdx, ctx, 0, comm, stream);

// 4. 等信号
ncclWaitSignal(nDesc, signalDescs, comm, stream);
```

### 内部两条路径

`scheduleRmaTasksToPlan` 根据 peer 是否在同一个 LSA (Look-Aside symmetric address space) team 内，决定走哪条路：

```
putSignal(peer)
     │
     ├── peer in same LSA ──→ CE 路径
     │                        GPU D2D memcpy + wait-value/write-value
     │                        via cuStreamBatchMemOp
     │                        完全不需要网络
     │
     └── peer outside LSA ──→ Proxy 路径
                              GPU 写 readySeq (cuStreamBatchMemOp)
                              CPU proxy 线程读到后：
                                1. 调 RMA plugin->iput() / iputSignal()
                                2. 写 doneSeq 回 GPU 可见内存
                              GPU 等 doneSeq (cuStreamBatchMemOp WAIT_VALUE_GEQ)
```

两条路径可**并行执行**（用两条 CUDA stream + event 同步）。

**CE 路径数据结构** (`src/include/rma/rma_ce.h:36-53`):
```
signalsWin layout (每个 CE context):
  [0..nRanks-1]          non-graph 每 rank signal
  [nRanks]               non-graph 聚合 signal
  [nRanks+1..2*nRanks]   graph 每 rank signal
  [2*nRanks+1]           graph 聚合 signal
  [2*nRanks+2..3*nRanks+1] graph 每 rank ack
```

**Proxy 路径数据结构** (`src/include/rma/rma_proxy.h`):
```
ncclRmaPutSignalOp {
    srcOff, srcHandle, dstOff, dstHandle, size, targetRank,
    signal { mhandle, offset, val, op },
    request    // RMA plugin 返回的异步操作句柄
}

ncclRmaProxyDesc {
    type: PutSignal | WaitSignal | PutSignalGroup
    state: Init | Ready | InProgress
    targetRank, context
    waitSeq (GPU 通知 proxy 的 readySeq 期望值)
}
```

### RMA Plugin vtable (`src/include/plugin/rma/rma_v14.h`)

```c
ncclRma_v14_t {
    init, devices, getProperties,
    listen, connect, createContext,
    regMrSym, deregMrSym,
    iput,           // 单边 put
    iputSignal,     // put + 信号
    iget,           // 单边 get
    iflush,         // 刷写
    test,           // 查询完成
    rmaProgress,    // 可选进度函数
    ...
}
```

具体实现：`src/transport/net_ib/gin.cc` 中的 `ncclRmaIbProxy`。

---

## GIN 详解

### 概念

GIN 是 NCCL 内部的一种 GPU 发起的网络抽象层，使得 **GPU kernel 可以直接发起跨节点通信**而无需 CPU 介入控制面。它有三种后端：

| 后端 | GPU 直接写网卡？ | CPU 参与？ | 内部 plugin | 需要硬件 |
|---|---|---|---|---|---|
| **GIN Proxy** | 否 | CPU proxy 做实际传输 | 有 (`ncclGinProxy`) | 无特殊要求 |
| **GIN GDAKI** | 是 (DOCA GPUNetIO) | 否 (配置阶段除外) | 有 (`ncclGinIbGdaki`) | DOCA 兼容 NIC (ConnectX) |
| **GIN GPI** | 是 (GFD 硬件队列) | **完全无** | **无**（外部 NIC plugin） | 下一代 NVIDIA NIC |

GIN Proxy 是主力实现；GDAKI 是现有 DOCA 方案；GPI 是下一代纯 GPU 驱动的方案。

### 数据流 (GIN Proxy)

```
GPU kernel 内部:
    1. 构造 GFD (GPU Function Descriptor) —— 128 字节包
       包含 opcode (put/get/flush)、src/dst 内存句柄、偏移、大小、
       信号描述符、counter ID 等
    2. 用 st.global.wt (write-through cache-hinted) 写入 peer 的
      lock-free multi-producer GFD 队列
    3. 原子增加 produce index (PI)

CPU proxy 线程 (ncclGinProxyProgress):
    1. 轮询 consume index (CI) → 发现新 GFD
    2. 读取 GFD，解析操作
    3. 调用底层 RMA plugin 执行:
       - iput()  / iputSignal() — Put / Put+Signal
       - iget()                 — Get
       - iflush()               — Flush
    4. 完成后更新 GPU 可见的 counter
       （通过 GDR 直接写 GPU 内存）
    5. 增加 consume index (CI)

GPU kernel 等待:
    ncclGinApi_Wait → waitForGfdComplete(GFD_idx)
    → 轮询 CI >= GFD_idx
```

### GFD 格式 (`src/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h`)

```
ncclGinProxyGfd_t (128 字节):
    header: flags, opcode, ...
    srcGinHandle / dstGinHandle
    srcOffset / dstOffset
    size
    signalDesc: signalWindow, signalOffset, signalOp, signalVal
    counterId
    inlineData[16]  (小数据直接嵌在 GFD 里)
```

### Device API (`src/include/nccl_device/gin/gin_device_common.h`)

模板函数，编译期分发到对应后端：

```c
ncclGinApi_Put<backend>(ctx, coop, peer, dstWin, dstOff, srcWin, srcOff, bytes, signal, ...)
ncclGinApi_Get<backend>(ctx, coop, peer, remoteWin, remoteOff, localWin, localOff, bytes, ...)
ncclGinApi_Wait<backend>(ctx, ...)
ncclGinApi_Flush<backend>(ctx, ...)
ncclGinApi_PutValue<backend>(ctx, ...)  // 小数据直接内联
```

分发机制 (`ncclGinCallImpl`): 运行时根据 `backendMask` 查到哪个 backend 的 bit 被设，switch 到对应的模板特化。

### GIN Plugin vtable (`src/include/plugin/gin/gin_v14.h`)

```c
ncclGin_v14_t {
    init, devices,
    getGinProperties, getProperties,
    listen, connect, createContext,
    regMrSym, deregMrSym,
    ginProgress,     // 进度函数 (供 GIN Proxy 使用)
    queryLastError,
    finalize
}
```

注意：GIN plugin 不直接提供 put/get 操作——那些操作通过 **GFD 描述 + proxy 转发到 RMA plugin** 完成。GIN plugin 主要负责连接管理、内存注册和设备句柄的创建。

---

## GPI 详解

GPI（GPU Packet Interface）是 GIN 的**纯 GPU 驱动**后端，完全不需要 CPU 参与数据路径。GPU kernel 直接把 64 字节的 GFD 写入 NIC 硬件队列，NIC 直接解析执行。

### 核心特性

- **无 CPU 参与** — 没有 proxy 线程，没有 RMA plugin 调用
- **64 字节 GFD** — 比 GIN Proxy 的 128 字节更紧凑，8 个 segment 各 64 bit
- **两种提交方式** — thread MMIO store 或 TMA 引擎拷贝
- **三种资源分享模式** — EXCLUSIVE / CTA / GPU
- **NCCL 内部没有 host 端 GPI plugin** — 由外部 NIC 厂商的 `libnccl-gin.so` 提供

### GFD 格式 (`src/include/nccl_device/gin/gpi/gin_gpi_device_host_common.h`)

```
gpi_gfd_t (64 字节，8 × 8 字节 segment):

  [0] HEADER
       bit 0:    owner (pi >> log_depth)
       bits 8-15: op
       bits 16-23: op_flags
       bits 24-39: counter_id
       bits 40-55: signal_id

  [1] DATA_DST
       bit 0:    owner
       bits 1-31: PE (processing element = rank)
       bits 32-63: size

  [2] DST_MEM_HANDLE
       bit 0:    owner
       bits 16-31: dst_handle
       bits 32-63: signal_value_low

  [3] SRC_MEM_HANDLE
       bit 0:    owner
       bits 16-31: src_handle
       bits 32-63: signal_value_high

  [4] DST_MEM_HANDLE_OFFSET
       bit 0:    owner
       bits 1-63: dst offset

  [5] SRC_MEM_HANDLE_OFFSET
       bit 0:    owner
       bits 1-63: src offset

  [6] INLINE_DATA_LOW
       bit 0:    owner
       bits 32-63: inline data low 32 bits

  [7] INLINE_DATA_HIGH
       bit 0:    owner
       bits 1-63: inline data high 64 bits
```

### 支持的操作

**Data ops**:

| Value | 名称 | 含义 |
|---|---|---|
| 0 | `READ` | 远程内存读 |
| 1 | `WRITE` | 远程内存写 |
| 2 | `WRITE_SIGNAL_ADD` | 写 + 原子加到 signal |
| 3 | `WRITE_SIGNAL_COUNTED` | 写 + 计数 signal |
| 4 | `WRITE_INLINE` | 数据内联在 GFD 中的写 |
| 5 | `WRITE_INLINE_SIGNAL_ADD` | 内联写 + signal 原子加 |
| 6 | `WRITE_INLINE_SIGNAL_COUNTED` | 内联写 + 计数 signal |
| 7 | `AMO_ADD` | 远程原子加 |
| 8 | `PE_FLUSH` | 刷新 peer ordering |

**Control ops**（bit 7 of op 置位，表示 `GPI_GFD_OP_CTRL`）:

| Value | 名称 | 含义 |
|---|---|---|
| 0 | `COUNTER_RESET` | 计数器归零 |
| 1 | `COUNTER_RESET_NO_WRITEBACK` | 计数器归零（不写回） |
| 2 | `COUNTER_WRITEBACK` | 显式计数器写回 |
| 3 | `SIGNAL_RESET` | 信号归零 |

**Operation modifiers**（与 data op 按位或）:

| Flag | 含义 |
|---|---|
| `WITH_COUNTER_FLAG (1<<0)` | 关联 counter |
| `WITH_COUNTER_COUNTED (1<<1)` | 计数递增 |
| `WITH_COUNTER_WRITEBACK (1<<2)` | 写回 counter 值 |

### 提交方式 (Post Mode)

GPU kernel 把 GFD 写入硬件队列的两种方式，由模板参数选择：

- **`GPI_POST_MODE_THREAD`** — 一个 CUDA thread 用 128 位 MMIO store (`st.relaxed.sys.global.b128`) 每次写两个 segment。CUDA 12.3 以下的 fallback 用 `v2.b64`。
- **`GPI_POST_MODE_TMA`** — 用 TMA 引擎 (`cp.async.bulk.global.shared::cta.bulk_group`)，GFD 在 shared memory 中构建，通过 TMA 硬件拷贝到 global memory。需要 **sm_90+** (Hopper/Blackwell)。

TMA 路径是 `__CUDA_ARCH__ >= 900` 时自动启用，否则退化到 thread 路径。

### 资源分享模式

GPI channel 的使用方式决定了原子操作的级别：

| 模式 | 原子开销 | 使用场景 |
|---|---|---|
| `EXCLUSIVE` | 无（普通 load/store） | 单个 CUDA thread 独享 channel |
| `CTA` | `thread_scope_block` 原子 + `ld.relaxed.cta`/`st.relaxed.cta` | 一个 block 内共享 |
| `GPU` | `thread_scope_device` 原子 + `ld.relaxed.gpu`/`st.relaxed.gpu` | 整个 GPU 共享（默认） |

### 队列结构

lock-free multi-producer ring buffer，NIC 作为消费者：

```
struct Queue_t {
    uintptr_t* gpu_memic_ptr;   // ring buffer 基址 (GPU 可见)
    uint64_t pi_;               // produce index (GPU 写)
    gpi_ci_t* ci_;              // consume index 指针 (NIC 写)
    uint64_t ci_value_;         // CI shadow（减少 GPU 读取）
    uint32_t log_depth;         // 队列深度 = 2^log_depth
};

struct gpi_gpu_channel_t {
    gpi_counter_t* gpu_counter_ptr_;  // counter 数组
    gpi_signal_t* gpu_signal_ptr_;    // signal 数组
    Queue_t queue_;
};
```

**操作流程**:
1. 线程原子递增 `pi_`（一次 GFD 一个 slot）
2. 检查 `(pi - ci) <= size`，满则等 CI
3. 计算 slot 索引 `pi & (size-1)`
4. 在 GFD 所有 segment 的 bit 0 写入 owner 标签 `pi >> log_depth`
5. 用 MMIO store 或 TMA 将 GFD 写入 `queue.gpu_memic_ptr[idx * 64]`
6. NIC 消费后更新 `ci_` 指针

### Counter / Signal 机制

- **`gpi_counter_t`** — 64 位，64 字节对齐。低 56 位 = 成功计数；bit 62 = error flag；bit 63 = writeback pending flag
- **`gpi_signal_t`** — 128 位（`value` + `flags`），64 字节对齐。bit 0 of flags = `GPI_SIGNAL_COUNTED_FLAG`
- 两种 signal 类型：
  - **Indexed signals** (`NCCL_GIN_SIGNAL_TYPE_INDEXED`) — 通过 `signalId` 索引
  - **VA signals** (`NCCL_GIN_SIGNAL_TYPE_VA`) — 通过虚拟地址（window + offset）定位

Flush 机制：发送 `PE_FLUSH` GFD，附带 ticket 值；NIC 完成后递增 counter；GPU 在 counter >= ticket 之前忙等。

### GPI 后端分发机制

GIN device API 通过模板特化 + 运行时 switch 完成后端分发：

```cpp
// 1. 每个 API 按后端模板特化
template <> struct ncclGinApi_Put<NCCL_NET_DEVICE_GIN_GPI> { ... };

// 2. ncclGinCallImpl 运行时分发
switch (ctx.backend) {
  case NCCL_NET_DEVICE_GIN_GPI:
    return ApiFn<NCCL_NET_DEVICE_GIN_GPI>::call(ctx, ...);
  case NCCL_NET_DEVICE_GIN_PROXY:
    return ApiFn<NCCL_NET_DEVICE_GIN_PROXY>::call(ctx, ...);
  ...
}
```

优化：如果 `backendMask` 只设了一个 bit（singleton），用 `__popc(beMask - 1)` 直接算出 case 编号，编译器可以死代码消除其他路径。

`NCCL_GIN_BACKEND_MASK_ALL` 在编译期计算：
```c
((NCCL_GIN_PROXY_ENABLE ? 1u : 0u) << NCCL_NET_DEVICE_GIN_PROXY |
 (NCCL_GIN_GDAKI_ENABLE ? 1u : 0u) << NCCL_NET_DEVICE_GIN_GDAKI |
 (NCCL_GIN_GPI_ENABLE   ? 1u : 0u) << NCCL_NET_DEVICE_GIN_GPI)
```

GPI 的 enable 条件是 `CUDA_VERSION >= 12020 && __CUDA_ARCH__ >= 700`，即 CUDA 12.2+ + sm_70+。

### 最低版本要求

从 `src/gin/gin_host.cc:25`：

```c
const int gpiBackendMinVersions[] = {0, NCCL_VERSION(2, 30, 5)};
```

GPI backend version 0 要求 **NCCL 2.30.5** 以上。Proxy 和 GDAKI 有 version 0 和 1，GPI 目前只定义了 version 0。

### GPI vs Proxy vs GDAKI 对比

| | GIN Proxy | GDAKI | GPI |
|---|---|---|---|
| CPU 参与度 | 有 (proxy 线程) | 仅初始化 | **完全无** |
| 数据面路径 | GPU→GFD→CPU→RMA plugin→NIC | GPU→GDAKI→NIC | GPU→GFD→NIC |
| GFD 大小 | 128 字节 | — | **64 字节** |
| 内部 NCCL 提供 host plugin | 是 (`ncclGinProxy`) | 是 (`ncclGinIbGdaki`) | **否** (需外部 NIC plugin) |
| 与 RMA plugin 关系 | 依赖 RMA plugin | 独立 | **独立** |
| submit 方式 | st.global.wt write | GDAKI 库调用 | MMIO store 或 TMA |
| 所需 CUDA arch | 无特殊要求 | sm_70+ | sm_70+ (TMA 需 sm_90+) |
| 所需 NCCL 版本 | 2.30.3+ | 2.30.3+ | **2.30.5+** |

---

## 关系

### 共享的底层

RMA 和 GIN **共享同一条 IB 级传输实现**——`src/transport/net_ib/gin.cc` 同时提供了：

- `ncclRmaIbProxy` (RMA plugin vtable)
- `ncclGinIbGdaki` (GIN plugin vtable)

GIN Proxy 后端运行时**调用 RMA plugin 的 `iput`/`iputSignal`/`iget`/`iflush`**。

### 层次结构

```
用户码 (nccl.h)
  │
  ├── ncclPutSignal / ncclSignal / ncclWaitSignal
  │     │
  │     ├── CE 路径 (同 LSA): cuStreamBatchMemOp D2D ─── GPU only
  │     │
  │     └── Proxy 路径 (跨 LSA): GPU→readySeq→CPU→RMA plugin→CPU→doneSeq→GPU
  │
  ├── ncclAllReduce / ncclAllGather / ... (collectives)
  │     │
  │     └── Collective kernel (GPU)
  │           │
  │           ├── 同节点: NVLink/NVSwitch D2D
  │           │
  │           └── 跨节点: GIN Proxy (GFD→RMA plugin) 或 GDAKI/GPI
  │
  └── 共享底层:
        src/transport/net_ib/gin.cc
          ├── ncclRmaIbProxy ─── RMA plugin vtable
          └── ncclGinIbGdaki ─── GIN plugin vtable
```

### 关键区别

| | RMA | GIN |
|---|---|---|
| **谁调用** | 用户程序 `ncclPutSignal(..., stream)` | Collective kernel `ncclGinApi_Put(ctx, peer, ...)` |
| **控制流** | host 端 enqueue → GPU stream 执行 | GPU kernel 内发起 |
| **描述符** | host 端构建 (rma_proxy_launch.cc) | GPU kernel 内构建 GFD |
| **等待方式** | stream 同步 (wait-value in `cuStreamBatchMemOp`) | Counter poll / GFD CI poll |
| **谁处理完成** | CUDA driver (CE) 或 proxy (RMA plugin test) | Proxy (GIN) 更新 counter |
| **graph 模式** | 持久化 descriptor，graph-safe D2D ack | 持久化 GFD slot，graph capture 兼容 |

### 容易混淆的点

1. **"RMA Proxy 和 GIN Proxy 是两个不同的 proxy 线程"**
   - RMA Proxy：`src/rma/rma_proxy_progress.cc`，处理 RMA 用户 API 的跨节点 put/signal/wait
   - GIN Proxy：`src/gin/gin_host_proxy.cc`，轮询 GPU kernel 产的 GFD 队列，**调用的仍是 RMA plugin**
   - 它们是独立的 progress 循环，服务于不同的调用者

2. **RMA 的 Proxy 路径和 GIN Proxy 调的是同一个 `iput`**
   - RMA Proxy 直接调 RMA plugin->iput()
   - GIN Proxy 解析 GFD 后也调 RMA plugin->iput()
   - 但在 RMA 中，描述符是 host 构建的；在 GIN 中，描述符 (GFD) 是 GPU kernel 构建的

3. **GIN GDAKI ≠ GIN Proxy ≠ GIN GPI**
   - GDAKI 是 GPU 通过 DOCA GPUNetIO 直接写 NIC 描述符，完全绕过 CPU
   - GIN Proxy 是 GPU 写 128B GFD 到队列，CPU proxy 读取后转发到 RMA plugin
   - GPI 是 GPU 写 64B GFD 到硬件队列，NIC 直接消费，连 proxy 和 RMA plugin 都不需要
   - 三者都是 GIN 的 backend，通过 `backendMask` 在 device API 中透明切换

4. **GPI 不需要内部 NCCL host plugin**
   - GIN Proxy 和 GDAKI 在 `src/transport/net_ib/gin.cc` 有 NCCL 内部的实现
   - GPI **没有** NCCL 内部的 host plugin。host 端的连接设置由外部 NIC vendor 的 `libnccl-gin.so` 通过 `NCCL_GIN_PLUGIN` 加载

5. **GPI 的 GFD 和 GIN Proxy 的 GFD 是两种不同的格式**
   - GIN Proxy GFD: 128 字节，含 src/dst 内存句柄、偏移、大小、signal 描述、counter ID，CPU proxy 解析后调 RMA plugin
   - GPI GFD: 64 字节，8 segment，NIC 硬件直接解析。segment 用 owner bit 防冲突

---

## 源码地图

### RMA 文件

| 文件 | 作用 |
|---|---|
| `src/nccl.h.in` | 公开 API 声明 (ncclPutSignal, ncclCommWindowRegister, ...) |
| `src/include/rma.h` | 内部初始化接口 |
| `src/include/rma/rma.h` | ncclRmaArgs, ncclRmaState, scheduleRmaTasksToPlan |
| `src/include/rma/rma_ce.h` | CE 路径上下文、信号窗口布局 |
| `src/include/rma/rma_proxy.h` | Proxy 路径描述符、队列、PutSignalOp |
| `src/rma/rma.cc` | 主调度：拆分 CE/Proxy、并行启动 |
| `src/rma/rma_ce.cc` | CE 路径：cuStreamBatchMemOp D2D |
| `src/rma/rma_proxy.cc` | Proxy 连接生命周期、内存注册 |
| `src/rma/rma_proxy_launch.cc` | Proxy 描述符构建、graph 支持 |
| `src/rma/rma_proxy_progress.cc` | Proxy progress 线程：issue + test 循环 |
| `src/include/plugin/rma/rma_v14.h` | RMA plugin vtable 定义 |
| `src/transport/net_ib/gin.cc` | ncclRmaIbProxy (IB RMA plugin 实现) |

### GIN 文件

| 文件 | 作用 |
|---|---|
| `src/include/gin.h` | 内部初始化接口 |
| `src/include/gin/gin_host.h` | Host 端 GIN 状态、连接管理 |
| `src/include/gin/gin_host_proxy.h` | GIN Proxy backend host 声明 |
| `src/gin/gin_host.cc` | GIN 连接建立、progress 线程；定义 `gpiBackendMinVersions` (NCCL 2.30.5+) |
| `src/gin/gin_host_proxy.cc` | GIN Proxy host: 轮询 GFD、调 RMA plugin |
| `src/include/nccl_device/gin/gin_device_common.h` | Device API 模板函数、backend 分发、`NCCL_GIN_BACKEND_MASK_ALL` |
| `src/include/nccl_device/gin/gin_device_api.h` | GIN device API 聚合头文件，条件 include 各 backend |
| `src/include/nccl_device/gin/proxy/gin_proxy.h` | GPU 端 GIN Proxy: postGfd, waitForGfdComplete (128B GFD) |
| `src/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h` | GIN Proxy GFD 格式定义 |
| `src/include/nccl_device/gin/gpi/gin_gpi.h` | **GPI device 实现**: Put/Get/Flush/Wait/PutValue/ResetSignal/ResetCounter |
| `src/include/nccl_device/gin/gpi/gin_gpi_device_host_common.h` | **GPI 格式定义**: 64B GFD, Queue, Channel, opcodes, post modes, sharing modes |
| `src/include/plugin/gin/gin_v14.h` | GIN plugin vtable 定义 |
| `src/transport/net_ib/gin.cc` | ncclGinIbGdaki (IB GIN plugin 实现) |

注意：`gin_gpi.h` 和 `gin_gpi_device_host_common.h` 是 device-only，NCCL 内部无对应的 host plugin 实现。
