# NCCL `src/transport/net.cc` 概要速查

> **文件**：`src/transport/net.cc`（约 2133 行）
> **定位**：NET 传输层实现（vtable 注册名 `"NET"`）——GPU 不能直接驱动网卡，由 CPU 代理线程（proxy）驱动 NIC（IB / TCP / 插件，如 NVLS）完成数据搬运，本文件定义二者协同的完整协议。
> **一句话**：net.cc 定义了「GPU FIFO 环形缓冲 + proxy 网络状态机」这一 GDR 加速的跨机数据传输协议：主机负责握手与内存映射，proxy 负责借/还信用、isend/irecv/test/flush 的推进，使 kernel 侧无感地以流水线方式在以太网/IB 上收发数据。
> **配套详读**：[`net-cc-analysis.md`](net-cc-analysis.md)（源码详解）、[`nccl_transport_net_cc_lifecycle.md`](nccl_transport_net_cc_lifecycle.md)（生命周期与函数时序）、[`net-cc-beginner-guide.md`](net-cc-beginner-guide.md)（小白入门）、[`proxy-internals.md`](proxy-internals.md)

---

## 1. 核心数据结构

| 结构 | 作用 |
|---|---|
| `connectMap` | 可跨进程共享的「内存地图」：5 个内存 bank（host mem / dev mem / shared host+dev mem / GDRCOPY mem）+ 编码偏移。偏移高 3 位编码 bank（001/011/101/111），低 29 位是 bank 内偏移；`NCCL_NET_MAP_GET_POINTER` 据此解出 cpu/gpu 指针 |
| `sendNetResources` / `recvNetResources` | proxy 侧每条连接的资源集：netComm、各协议缓冲与 MR 句柄、GDR/flush 状态、step 计数等 |
| `ncclSendMem.head` | 发送端已确认步数（信用额度） |
| `ncclRecvMem.tail + connFifo[NCCL_STEPS=8]` | 接收端 FIFO：每槽 `{size, offset}`，全部放在 GPU 可见内存 |

---

## 2. 核心函数

### 主机侧（host）

| 函数 | 功能 |
|---|---|
| `canConnect` | 不同主机必走 NET；同主机由 `ncclTopoCheckNet` 决定（可禁 intra-node net） |
| `sendSetup` / `recvSetup` | 选 NIC（`ncclTopoGetNetDev`）、判定 GDR（`ncclTopoCheckGdr` → `NCCL_DIRECT_NIC`）与 flush 需求、确定 proxy rank（PXN 时发送可走远端 proxy）；recv 侧 `listen()` 生成 `ncclNetHandle_t` 放入 `connectInfo`（bootstrap 交换，另附 useGdr 标志） |
| `sendConnect` / `recvConnect` | **异步建连**：首次 `ncclProxyCallAsync` 投递请求，之后 `ncclPollProxyResponse` 轮询至 proxy 填好 `connectMap`；随后主机将 map 内存导入本进程地址空间（跨进程 SHM import / CUDA IPC import，必要时 `cudaDeviceEnablePeerAccess`），把 `head/tail/connFifo/buffs` 指针写入 `conn`，使 kernel 与 proxy 指向同一 FIFO |
| `sendFree` / `recvFree` | 逆序释放 IPC/SHM 映射与 map |

### Proxy 侧（真正的数据通路）

| 函数 | 功能 |
|---|---|
| `sendProxySetup` / `recvProxySetup` | 建资源、读网卡属性（maxRecvs、maxP2pBytes、DMA-BUF 能力） |
| `sendProxyConnect` / `recvProxyConnect` | 插件 `connect`/`accept` 建真实网络连接；分配各协议缓冲（LL 用 host 内存，LL128/SIMPLE 用 GDR 设备内存，DMA-BUF 或 nv_peermem 注册 MR）；可选 GDRCOPY（`gdcSync` 同步字 + `gdcFlush`）；可选共享缓冲池/共享连接（`NCCL_NET_SHARED_BUFFERS/COMMS`）；最终把 `connectMap` 回传主机 |
| **`sendProxyProgress`** | 发送状态机，三段：① 给 GPU 发信用（写 `head`）；② 等 GPU 填满 FIFO 槽（LL/LL128 校验每行 flag）后调 `isend`；③ `test()` 完成即回写 `head`，GPU 回收槽位 |
| **`recvProxyProgress`** | 接收状态机，四段：① 批量 `irecv`（同 recvComm 的 sub 归组**多路合投**，LL 可忽略完成事件）；② `test()` 完成 → 置槽有效；③ GDR 场景 flush（GDRCOPY 用一次 PCIe 读 `gdcFlush` 强制 NIC posted write 落点，否则插件 `iflush`），回写 `tail`；④ 等远端 `head` 推进后回收缓冲 |
| `netRegisterBuffer` / `ncclNetLocalRegisterBuffer` / `ncclNetGraphRegisterBuffer` / `*ProxyRegBuffer` | RMA 用户缓冲注册：主机经 `ncclProxyCallBlocking` 请求 proxy，proxy 用 DMA-BUF（支持多物理段）或 `regMr` 注册并回传 handle；句柄挂 `proxyMemHandleQueue`，销毁时统一 `deregMr`；CUDA graph 场景额外挂 `cleanupNet` 回调 |
| `sendProxyFree` / `recvProxyFree` | 逆序释放 MR、连接、共享缓冲池、SHM/IPC 映射 |

---

## 3. 机制与实现原理

1. **流水线 + 信用制（credits）**：每次传输切为 `NCCL_STEPS=8` 个槽的环形 FIFO。发送方 GPU 写满槽置 `size`，proxy 见 `size != -1` 才 `isend`；完成后 proxy 推进 `head`，GPU 才能复用槽。槽深有限（8，共享模式 16）天然限流；共享模式初始 `head = -NCCL_STEPS` 扣发信用，防死锁。
2. **GDR（GPU Direct RDMA）**：SIMPLE/LL128 缓冲放设备内存，网卡直接 DMA 进出 GPU 内存（`regMr`/nv_peermem 或 `regMrDmaBuf`），避免 CPU 中转；LL 数据小，放 pinned host 内存。`useGdr==Pci` 时强制 DMA-BUF 走 PCIe 映射（`CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE`）。
3. **GDRCOPY 优化**：`head`/`tail` 同步字放 GPU 内存、经 gdrcopy 映射到 CPU，proxy 用普通存储 + `wc_store_fence` 更新，省掉 cudaMemcpy；flush 用一次 PCIe 读强制冲刷 NIC 的 posted write。
4. **跨层指针一致性**：FIFO/缓冲地址经 `connectMap` 同时暴露给主机（导入后 GPU 用）与 proxy（CPU 用），两视图指向同一物理内存。
5. **连接复用与 PXN**：发送端 proxy 可位于远端 rank 进程（PXN）；共享模式下多 channel/rank 复用同一 netComm 与缓冲池，减少连接与 MR 数量。
