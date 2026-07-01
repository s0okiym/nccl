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

| 后端 | GPU 直接写网卡？ | CPU 参与？ | 需要硬件 |
|---|---|---|---|
| **GIN Proxy** | 否 | CPU proxy 做实际传输 | 无特殊要求 |
| **GIN GDAKI** | 是 (DOCA GPUNetIO) | 否 (配置阶段除外) | DOCA 兼容 NIC |
| **GIN GPI** | 是 | 否 | 支持 GPI 的 NIC |

GIN Proxy 是主力实现，其余两个是高级优化。

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

3. **GIN GDAKI ≠ GIN Proxy**
   - GDAKI 是 GPU 直接写 NIC 描述符，完全绕过 CPU
   - GIN Proxy 是 GPU 写 GFD 到队列，CPU proxy 读取后转发
   - 两者都是 GIN 的 backend，通过 `backendMask` 在 device API 中切换

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
| `src/gin/gin_host.cc` | GIN 连接建立、progress 线程 |
| `src/gin/gin_host_proxy.cc` | GIN Proxy host: 轮询 GFD、调 RMA plugin |
| `src/include/nccl_device/gin/gin_device_common.h` | Device API 模板函数、backend 分发 |
| `src/include/nccl_device/gin/gin_device_api.h` | GIN device API 聚合头文件 |
| `src/include/nccl_device/gin/proxy/gin_proxy.h` | GPU 端 GIN Proxy: postGfd, waitForGfdComplete |
| `src/include/nccl_device/gin/proxy/gin_proxy_device_host_common.h` | GFD 格式定义 |
| `src/include/plugin/gin/gin_v14.h` | GIN plugin vtable 定义 |
| `src/transport/net_ib/gin.cc` | ncclGinIbGdaki (IB GIN plugin 实现) |
