# NCCL EP（Expert Parallelism）实现深度解析

> 本文聚焦 NVIDIA NCCL 的 Expert Parallelism 扩展，而不是普通 NCCL collective 中的某个 EP 算法。当前仓库 `nccl/` 是 NCCL 2.29.7，尚无 `contrib/nccl_ep`；本文审阅对象是 NVIDIA/nccl 官方上游提交 [`5067397c`](https://github.com/NVIDIA/nccl/tree/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep) 中的 `contrib/nccl_ep`，其库版本为 0.1.0、API version 为 1。后续若代码合入本仓库，应先重新核对差异。

---

## 1. 先给出结论

NCCL EP 是独立的 `libnccl_ep`，提供 MoE 专用的 `ncclEpDispatch` 与 `ncclEpCombine`。它复用 NCCL communicator 的拓扑、LSA team、rail team、对称内存窗口和 Device API，但 token 数据面不经过普通 collective 的 task/planner、算法/协议选择和通用 collective kernel；HT 路由预处理中的 `ncclAllGather` 仍是一次普通 NCCL collective。其余 EP 数据面由自己管理路由元数据、缓冲区、kernel 和完成协议。

它不是一个算法，而是两套共享 API 的后端：

| 后端 | 设计目标 | 主要数据通路 | 典型场景 |
|---|---|---|---|
| Low-Latency（LL） | 最小化单步时延 | 节点内 LSA 直接写；节点间 GIN point-to-point put | decode、小 batch、对尾延迟敏感的推理 |
| High-Throughput（HT） | 最大化大批量吞吐 | 节点内分发/归约 + 节点间分层 GIN RDMA；TMA 流水 | 训练、prefill、大 batch、高并发 token |

算法由调用者在 `ncclEpGroupConfig_t.algorithm` 中明确选择；代码中没有运行时 cost model 或 LL/HT 自动切换。所谓“自动配置”目前只是若干默认值和 LL buffer 自动扩容，不是基于测量的 autotuning。

最重要的实现思想是：**先把稀疏的 token→expert 路由转成稳定的通信映射，再让 GPU 直接发起 NVLink/网络传输，并把 MoE 的去重、复制、压缩与归约语义融合进通信 kernel。**

---

## 2. EP 要解决的通信问题

设：

| 符号 | 含义 |
|---|---|
| `R` | communicator 中的 rank 数 |
| `N` | 节点/LSA domain 数，即 HT 的 rail team 大小 |
| `G` | 每个 LSA team 的 rank 数 |
| `E` | 全局 expert 数 |
| `L = E/R` | 每个 rank 固定持有的本地 expert 数 |
| `B` | 当前 rank 的源 token 数 |
| `K` | top-k |
| `H` | hidden dimension |
| `Msend/Mrecv` | 每 rank 配置的最大发送 token 数/接收 slot 数 |

router 给出 `topk_idx[b,k] = e` 和权重 `w[b,k]`。当前实现按连续编号放置 experts：`owner_rank(e)=floor(e/L)`，`local_expert(e)=e mod L`。forward 的逻辑结果为：

```text
dispatch: x[b]  →  owner(e) 上的 expert e
expert:   z[b,k] = F_e(x[b])
combine:  y[b] = Σ(k=0..K-1) w[b,k] · z[b,k]
```

直接用通用 All-to-All 实现会暴露几个额外阶段：计算每个 peer 的计数和 offset、打包、交换大小、传输、解包、按 expert 重排，以及 combine 的逆向索引和归约。路由又会逐步变化，负载也可能严重不均。NCCL EP 的设计初衷是把这些阶段变成一个长期复用的 EP group、一个绑定路由的 handle，以及专门的 fused kernel，减少 CPU 参与、kernel 次数和中间搬运。

一个关键优化是不按 expert 重复发送相同 token：

- LL 对同一 token 的 top-k expert 按目标 rank 去重，线缆上每个目标 rank 最多发送一次；目标 rank 再在本地复制到相应 expert。
- HT 跨节点时按目标 node 去重，token 每个目标 node 最多经 NIC 发送一次；到达节点后再通过 LSA 分发到目标 rank。FLAT layout 在一个目标 rank 内也只保留一个 token slot。

---

## 3. 实现分层与源码地图

```text
应用 / MoE framework
  │  ncclEpCreateGroup / CreateHandle / Dispatch / Combine
  ▼
nccl_ep.cc：ABI、参数检查、资源生命周期、host 编排
  ├─ Handle preprocessing：routing bitmap + NCCL AllGather + scan/remap
  ├─ LL adapter ──────────► low_latency.cu（静态编译 kernel）
  └─ HT adapter ──────────► hybridep_adapter.cu
                              ├─ runtime JIT：scan/dispatch/combine
                              └─ hybrid_ep.cuh：TMA/warp-specialized pipeline
  ▼
NCCL Device API
  ├─ LSA：同一 load/store domain 内的对端显存访问
  └─ GIN：GPU 发起 put、signal、barrier 等网络操作
  ▼
NVLink / NVSwitch / NIC RDMA
```

| 层次 | 主要文件 | 职责 |
|---|---|---|
| 公开 ABI | `include/nccl_ep.h`、`include/ep_enums.h` | tensor、group、handle、layout、dispatch/combine API |
| Host runtime | `nccl_ep.cc`、`include/common.hpp` | 参数、buffer、window、Device Comm、路由预处理、调用编排 |
| LL | `device/low_latency.cu` | 延迟优先的 direct/GIN dispatch 与 combine |
| HT host adapter | `device/hybridep_adapter.cu[h]` | kernel 参数、shared-memory 计算、稀疏/稠密权重转换 |
| HT device | `device/hybrid_ep.cuh`、`device/device_primitives.cuh` | TMA pipeline、GIN、层次化分发/归约、同步 |
| Metadata | `device/scan_kernel.cuh` | bitmap 扫描、slot 压缩、expert-major remap |
| Runtime JIT | `device/jit/*` | 生成源代码、NVCC 编译、cubin/cache/module/launch |

LSA 和 GIN 的语义可分别参考 NCCL 官方的 [Device API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device.html) 与 [GIN API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/device_gin.html)。EP 的特殊之处不只是“使用 Device API”，而是让专用 GPU kernel 同时负责路由读取、数据搬运、信号发布、等待和归约。

`common.hpp`、LL 和 HT 源文件都声明部分实现改编自 DeepEP。NCCL EP 保留了按 MoE 语义定制的低延迟/层次化 pipeline，同时把拓扑发现、对称 window、节点内 peer pointer 和 GPU 网络操作接到 NCCL Device API 上，并在其外建立独立 C ABI。

---

## 4. API 对象、所有权与生命周期

### 4.1 推荐生命周期

```text
已有 ncclComm_t
   │
   ├─ ncclEpCreateGroup                 全 rank 同序调用，建立长期资源
   │
   ├─ ncclEpHandleMemSize（可选）
   ├─ ncclEpInitHandle                  绑定 layout/K、分配映射空间
   ├─ ncclEpUpdateHandle                绑定本步 routing；HT 是 collective
   │      或 ncclEpCreateHandle = Init + Update
   │
   ├─ ncclEpDispatch
   ├─ 本地 expert 计算
   ├─ ncclEpCombine
   ├─ 重复 UpdateHandle/Dispatch/Combine
   │
   ├─ ncclEpHandleDestroy               必须在 group 前
   ├─ ncclEpGroupDestroy                必须在基础 comm 前
   └─ ncclCommDestroy
```

`ncclEpGroup` 持有拓扑相关、所有 handle 共享的昂贵资源；`ncclEpHandle` 持有 layout、K、当前路由的反向映射和 slot metadata。把两者分开，是为了让 communicator/window/GIN setup 跨多步复用，同时允许路由逐步更新。

### 4.2 ABI 防误用机制

所有跨 ABI 的配置结构都以 `size` 和 `magic` 开头，group config 还带 `version`。调用者必须使用对应的 `NCCL_EP_*_INIT` 初始化宏。当前实现要求 `size == sizeof(struct)`，并不具备结构体尾部扩展的前向兼容性；version 不匹配只打印警告，size/magic 不匹配则触发断言。

`ncclEpTensor_t` 只是 descriptor：

- `ncclEpTensorAlloc` 只分配 descriptor 和复制后的 `sizes[]`，**不分配 payload**；
- `ncclEpTensorDestroy` 也只销毁动态 descriptor，不释放 `data` 或 window；
- 栈上静态 descriptor、`sizes[]`、普通 device pointer 和外部 window 均由调用者持有；
- tensor 可用普通 `data`，也可用 `win_hdl + win_offset`。外部 window 允许 HT kernel 直接读写用户注册内存；
- 任一维为 0 时允许 `data == nullptr`，作为合法空 tensor。

### 4.3 关键 API 的实际语义

| API | 实际行为与注意点 |
|---|---|
| `ncclEpCreateGroup` | 内部创建并同步私有 CUDA stream；分配/register 长期资源；是 collective |
| `ncclEpHandleMemSize` | LL、HT 的源码都要求 `num_topk > 0`，尽管 header 对 HT 写着 optional |
| `ncclEpInitHandle` | LL 的 AUTO RDMA buffer 首次分配或扩容时实际是 collective；HT 本地分配 handle block |
| `ncclEpUpdateHandle` | LL 保存 routing descriptor；HT 缓存 routing payload 并 AllGather 全局 bitmap |
| `ncclEpCreateHandle` | 从 `topk_idx.shape[1]` 得到 K，因而比手工 `InitHandle(...,-1,...)` 更符合当前源码 |
| `ncclEpDispatch/Combine` | 输出均由调用者预分配；API 不替用户执行 expert MLP |
| `ncclEpComplete` | 仅补做 LL `send_only` 的 receive phase；HT 是 no-op |
| `ncclEpGroupDestroy` | 先 `cudaDeviceSynchronize()`；LL 还执行跨 rank barrier 后 deregister/free |

实践上，HT 也应始终向 `HandleMemSize/InitHandle` 传真实 `K > 0`。header 中“HT 可传 `-1`”与实现的 `assert(num_topk > 0)` 不一致。

---

## 5. Group：拓扑、配置和长期资源

### 5.1 通用不变量

创建 group 时至少要求：

- communicator 有效且 `deviceApiSupport=true`；
- `E > 0`、`Msend > 0`、`max_token_bytes > 0`；
- 最终在 handle 初始化时要求 `E % R == 0`，即每 rank 专家数相等；
- 所有 rank 的 `Mrecv` 相同；HT 要求 `Mrecv >= Msend > 0`；
- HT 的 `Msend <= MAX_SUPPORTED_TOKENS_PER_RANK=8192`；
- 自定义 `alloc_fn/free_fn` 必须成对提供。

自定义 allocator 只覆盖 workspace、部分 metadata、counter 和 handle block；LL/HT 对称通信区仍通过 `ncclMemAlloc/ncclMemFree` 管理，部分控制对象也直接使用 CUDA allocator。因此它不是所有 EP 显存分配的统一替换点。

`max_token_bytes` 是 dispatch 和 combine 单个 token 的统一上界，不等于 hidden dimension。每次调用的 `H × dtype_size` 仍会单独检查。

默认配置的实际解析为：

| 配置 | `NCCL_EP_AUTO` 的当前行为 |
|---|---|
| `max_recv_tokens_per_rank` | LL 变成 `R × Msend`；HT 不允许 AUTO |
| `rdma_buffer_size` | LL 在首次 handle 初始化时按 layout/K 延迟分配，并可在后续 handle 上 collective 扩容 |
| `max_num_sms` | HT 为 16；LL 为设备全部物理 SM |
| `num_qp_per_rank` | `max_num_sms × HYBRIDEP_DISPATCH_N2N_WARPS`，当前常量为 2 |
| `num_channels` | 置为 10；之后未被 host 或 device 数据路径读取 |

超时优先级是 `NCCL_EP_TIMEOUT_MS` 环境变量、`timeout_ns`、编译期约 100 秒默认值。`timeout_ns` 会先整除为毫秒，亚毫秒配置会被截断。

### 5.2 Team 分解

EP 直接取：

- `ncclTeamLsa(comm)`：同一 load/store-accessible domain，通常是节点内 NVLink/NVSwitch 域；
- `ncclTeamRail(comm)`：HT 的跨 LSA domain 同 rail 通信队列；
- HT 强制 `rail_team.nRanks × lsa_team.nRanks == R`。

代码还通过 hostname AllGather 统计物理节点，并按 `R/nNodes` 推导 `gpus_per_node`、`rank_in_node` 和 `node_id`。HT 的热路径主要使用 NCCL team rank，但 hostname 结果仍影响单/多节点初始化和部分 handle 分配；因此当前实现最适合各节点 GPU 数一致、EP ranks 按常规方式组成 LSA/rail teams 的对称部署。

### 5.3 LL group 资源

LL 在 world communicator 上创建一个 `ncclDevComm`：

- 只有一个 LSA domain 时，可以完全走 LSA direct path，不要求 GIN；
- 多 LSA domain 时要求 GIN、full connection、`num_qp_per_rank` 个 context；
- signal 空间包含 combine、dispatch 和一个 clean barrier；
- RDMA 对称 buffer 可在 group 创建时显式分配，也可在 handle 初始化时延迟分配；
- 开启 mask 时另建一个独立的对称 sync buffer，避免 LL 主 buffer 扩容使恢复 barrier 失效；
- 所有模式另有 32 MiB `ep_workspace`。

虽然 group 会按 AUTO 规则申请较多 GIN contexts，LL kernel 的当前数据路径始终使用第 0 个 Device Comm，并把 expert/rank hash 映射到前 4 个 contexts（`MAX_NCCL_GIN_CTX_PER_COMM=4`）；把 `num_qp_per_rank` 配到 4 以上目前不会让 LL kernel 使用更多 context。

LL handle 只保存 buffer-relative offset。因此 AUTO 扩容时可以 barrier、deregister、free、重新分配/register 主 buffer，而既有 handle 仍能用新 base 加旧 offset；代价是扩容期间所有 rank 必须同序进入，且不能有在途 EP 操作或已捕获图继续引用旧 base。

### 5.4 HT group 资源

HT 的节点内资源包括：

1. 一个 `ncclMemAlloc` mega buffer，依次容纳 dispatch token/probability 和 combine token/probability staging；
2. 一个单独注册的 completion flag window；
3. host-pinned 的 LSA peer pointer 数组；
4. 每 rank 32 字节 generation/counter block，含 dispatch/combine 的期望 RDMA 值、期望节点内完成值和两个 grid barrier counter；
5. group 级全局 routing bitmap：`R × Msend × ceil(E/8)` 字节。

多节点时再分配一个 4 KiB 对齐的大型 GIN buffer，内部切成：

- 节点内预归约结果和远端 combine 结果；
- dispatch/combine chunk flags；
- token、dense probability、scale staging；
- packed RDMA send/receive 区域。

随后创建一个 rail-connected Device Comm。queue depth 为 `3 × 64 + 1 = 193`，GIN context 数至少为 `2 × max_num_sms`。这些资源是 group 共享的，不随 handle 数量复制。

---

## 6. Handle 与 HT 路由预处理

### 6.1 Handle 内存

调用者可以先用 `ncclEpHandleMemSize` 查询大小，再传入一块 1D `ncclUint8` device tensor；否则库通过 group allocator 分配。

LL handle 主要包含：

- 每个源 rank 的接收计数及 `(token_id, K 个源 slot)` 映射；
- `[L,R]` 的 expert dispatch range/layout；
- 当前 layout/K 对应的 LL buffer offsets；
- 一份 `topk_idx` descriptor 的永久拷贝。其 `sizes[]` 被复制，但 payload 仍属调用者，必须在使用期间有效。

HT handle 主要包含：

| 映射 | 作用 |
|---|---|
| `rdma_to_attn_map` | 当前节点应从各远端 node 接收哪些 token |
| `attn_to_rdma_map` | 本节点哪些 token 要发送到各远端 node |
| `token_rank_mask` | 每个源 token 在目标节点内需要哪些 LSA ranks |
| `sparse_to_dense_map` | 源 token 到接收 compact slot 的映射，combine 反向复用 |
| `local_expert_routing_map` | 每个接收 slot 对应当前 rank 的哪些本地 experts |
| `num_tokens_for_experts` | 当前 rank 实际接收 slot 总数 |
| cached `topk_idx` | HT 自有的 routing payload 副本，保持 K-slot 原始顺序 |
| EM counts/offsets | expert-major 的实际计数、对齐后 zone offset |

单节点 HT 的 dense probability `[Msend,E]` 在 handle block 内；多节点时复用 group GIN buffer 中的同类区域。

### 6.2 `ncclEpUpdateHandle` 的三步流程

HT 每次路由更新都是 collective，并在调用 stream 上依次执行：

```text
topk_idx[B,K]
   │  convert_topk_to_routing_map；同时清零 [B,Msend) 尾部
   ▼
本 rank bitmap [Msend, ceil(E/8)]
   │  ncclAllGather(base communicator)
   ▼
全局 bitmap [R,Msend,ceil(E/8)]
   │  JIT metadata scan/remap
   ▼
node/rank masks + compact slot + local expert map + counts/offsets
```

尾部清零不是优化细节：AllGather 总是发送 `Msend` 行；如果本步 `B` 变小，旧路由 bit 会被其他 rank 当成有效 token。`test_ht_stale_routing_map.cu` 专门覆盖这个问题。

`topk_idx` 的当前 host 路径没有逐项检查范围或唯一性。HT bitmap kernel 对任意 `expert >= 0` 直接访问 `row[expert/8]`，没有验证 `expert < E`；越过 `ceil(E/8)` 可造成越界写。同一 token 的重复 expert 还会使 dense-weight scatter 出现重复写。调用者必须保证每项为 `-1` 或 `[0,E)` 内的互异 expert id，且 HT 遵守公共 `K<=32`、LL 遵守实际 `K<=9`。

### 6.3 FLAT 与 EXPERT_MAJOR 的 slot 生成

FLAT 以“目标 rank 上的唯一源 token”为 slot：一个 token 即使命中该 rank 的多个本地 expert，也只占一个 slot；`local_expert_routing_map[slot,*]` 记录其 expert 子集。`sparse_to_dense_map` 大致按 `[source_node, token, destination_lsa_rank]` 建立源→目标 slot 映射。框架应在目标 rank 上运行这些本地 experts，并把同 rank 的结果预归约为一个 slot，再交给 combine。

EXPERT_MAJOR 以 `(源 token, 本地 expert)` 为 slot，因此同一 token 命中多个本地 expert 会被复制多次。remap kernel：

1. 计算每个本地 expert 的实际 token 数；
2. 按 `dispatch_output_per_expert_alignment`（必须是 2 的幂）向上取整每个 expert zone；
3. 生成 zone offset 和 packed source→destination 映射；
4. dispatch 的 PAD warp 将 padding slot 清零。

EM remap 当前最多支持每 rank 64 个本地 expert。若 FLAT 实际 slot 或 EM 对齐后的总 slot 超过 `Mrecv`，device 端会 trap，而不是丢弃 token 或自动扩容。

`layout_info` 中的 `expert_counters`、`expert_offsets` 和 `recv_total_counter` 在 **UpdateHandle 的 metadata kernel** 中生成，不是在 dispatch 后才计算。它们可用于选择有效 expert 区间，但当前仍必须事先配置静态 `Msend/Mrecv`。

---

## 7. 四种 layout 的完整语义

| 后端/layout | Dispatch token 输出 | Dispatch 路由输出 | Expert 侧责任 | Combine 权重责任 |
|---|---|---|---|---|
| LL / EXPERT_MAJOR | `[L, R×Msend, H]` | `expert_counters[L]`；无 per-slot idx/weight | 每个 expert 处理自己的 zone | home rank 的 combine 根据原始 `[B,K]` 权重逐 expert 加权 |
| LL / RANK_MAJOR | `[R,Msend,H]` | weights `[R,Msend,K]`、local idx `[R,Msend,K]`、source counts `[R]` | 运行 slot 命中的本地 experts并先加权归约成一个向量 | combine 每个 expert rank 只返回一个已归约贡献，接收端权重为 1 |
| HT / FLAT | `[Mrecv,H]`，有效前缀由 metadata 给出 | weights `[Mrecv,K]`、local idx `[Mrecv,K]` | 同 rank 多 expert 结果先按权重归约成一个 slot | HT 只跨 rank/node 求和，不再应用 forward 权重 |
| HT / EXPERT_MAJOR | `[Mrecv,H]`，内部按 expert zone 排列 | scalar weight `[Mrecv]`、counts/offsets；无 idx | 每个 slot 对应一个 expert；调用者先乘 scalar weight | HT 对所有已加权 expert contribution 求和 |

LL EXPERT_MAJOR 有一个容易误读的 ABI：原始 routing weights 通过 `ncclEpCombineOutputs_t.topk_weights` 传入，虽然字段属于 `outputs`，实现却把它当作 combine kernel 的只读权重输入。它必须是 `[B,K] float32`。这是当前 API 设计，不应根据结构名推断数据方向。

HT 的 forward/backward 由 `pass_direction` 明确选择：

| 操作 | FWD | BWD |
|---|---|---|
| Dispatch 输入 weights | 必须有 `[B,K]` | 必须为 NULL，复用 handle routing |
| Dispatch 输出 weights/idx | FLAT 或 EM 按上述 layout 输出 | 必须为 NULL |
| Combine 输入 weights | 必须为 NULL；expert 结果已应用 forward 权重 | 必须提供：FLAT `[Mrecv,K]`，EM `[Mrecv]` |
| Combine 输出 weights | kernel 不使用 | 输出 `[B,K]`，按 cached `topk_idx` 恢复原 K-slot 顺序 |

LL 只接受 FWD；传 BWD 会返回 `ncclInvalidUsage`。

---

## 8. LL：延迟优先实现

### 8.1 Buffer 与 wire format

每个 LL group 只有一块对称 RDMA buffer。`LowLatencyLayout` 为每个 handle 计算 offset，并使用两个 odd/even ping-pong slot。一个 slot 在 dispatch 与 combine 间复用，因为实现假定二者不会并发；slot 大小取两种操作 `send + receive` 的较大值。每个 slot 另有共享的 dispatch receive-count/combine receive-flag 区域；当前操作运行时顺便清理下一个 slot。

Dispatch message 包含：

```text
header = align16(token_id:int32 + K × router)
router(EXPERT_MAJOR) = expert_id:uint16
router(RANK_MAJOR)   = {weight:float32, expert_id:uint16, padding}  // 8 B
payload = BF16 token，或 dispatch 内部量化后的 FP8 token + scales
```

节点内 direct path 为了便于并行 load/store，把接收区分成“所有 headers + 所有 payloads”；GIN RDMA 路径使用连续的 `header|payload|scales` message。接收 kernel 根据 transport 选择相应布局。

Combine message 主要是 token payload 和内部预留的 scale metadata。当前公开 combine 调用固定 `use_fp8=false`、`zero_copy=false`，所以 end-to-end combine 是 BF16。

### 8.2 Dispatch 数据路径

LL kernel 按 expert 分配 warp group/SM，执行以下流程：

1. warp 读取 `topk_idx[B,K]`；对命中同一目标 rank 的 K 项做 warp-level 去重；
2. 统计每个目标 rank/token 数，并为发送 slot 分配 offset；
3. 写入 token_id、目标 expert id，RANK_MAJOR 还写 routing weight；
4. 若目标在同一 LSA domain，通过 `ncclGetP2pPtr` 得到对端 window 地址并直接写；否则从已注册 staging 用 `ncclGin::put` 发往远端；
5. payload 完成后发布 count：P2P 使用 system-scope release store 写 `-count-1`，GIN 使用 0-byte put + `SignalAdd(count+1)`；
6. 接收侧轮询 count/signal，超时检查后读取 payload；
7. EXPERT_MAJOR 对每个匹配的本地 expert 原子分配 slot，并缓存该 slot 的源 token/K 位置；RANK_MAJOR 写入 source-rank 固定区域及 per-slot weights/local expert ids。

“先 payload、后 count/signal”是 LL 的 ready 不变量。count 不只是数量，也承担发布数据可见性的 generation/完成标志。节点内与网络路径采用相反符号编码，是为了在统一接收逻辑中区分和还原数量。

### 8.3 Combine 数据路径

Combine 复用 dispatch 留下的 source mapping，沿反方向发送 expert 输出：

- EXPERT_MAJOR：每个 expert contribution 单独返回；home rank 用原 `topk_idx/topk_weights` 选择并加权，FP32 累加后写 BF16；
- RANK_MAJOR：目标 rank 上的框架已经把多个本地 expert 结果按权重预归约；kernel 对同一 home token/目标 expert rank 再去重，返回一个权重为 1 的 contribution；
- 同一 LSA domain 直接写对端 combine receive buffer；跨 domain 使用 GIN put；
- sender 在数据后发布 per-expert flag，receiver 等待所有未 masked contribution，再用 TMA/向量加载和 FP32 reduction 生成 `[B,H]`。

这解释了两种 layout 的权衡：EXPERT_MAJOR 让 expert 计算最直接，但 combine message 数可能更高；RANK_MAJOR 将本 rank 的多 expert 归约前移，降低返回流量，却把本地重排和预归约责任交给框架。

### 8.4 `send_only` 与 `Complete`

LL dispatch/combine 可先以 `send_only=1` 只启动 send phase，稍后 `ncclEpComplete` 再发射 receive phase：

```text
EP send kernel → 可重叠的本地工作 → EP receive kernel
```

每个 handle 只有一个 `continue_fn`；在完成前再次发起 staged 操作会覆盖它，源码没有队列或保护。`Complete` 的 `stream` 参数当前未参与 launch，lambda 使用的是原 dispatch/combine 调用时捕获的 stream。因此应在同一 handle 上严格保持“一次 staged op → 一次 Complete”，并显式管理跨 stream 依赖。

### 8.5 LL 的硬约束

- 输入和 combine 均为 BF16；hidden 必须是 `2048/2560/4096/5120/6144/7168/8192` 之一；
- host 还要求 hidden 为 128 的倍数、token bytes 不超过 group 上界；
- 公共常量 `MAX_NUM_TOPK=32`，但 LL launch 实际断言 `K <= 9`；
- `Msend × L` 必须为 4 的倍数；
- warp-group 当前最多 14 组，因此过小的 `max_num_sms` 会被拒绝；
- dispatch 提供 scales 时存在 BF16→FP8 的量化分支，但 combine 固定 BF16，不能据此认定完整 FP8 EP 已受支持。

---

## 9. HT：吞吐优先的层次化实现

### 9.1 为什么要分层

假设同一 token 命中一个远端节点上的多个 experts。逐 expert RDMA 会重复发送 token；逐 rank 发送也可能在节点内重复使用 NIC。HT 将网络域与 LSA 域分开：

```text
源节点：token 按目标 node 压缩并发送一次
目标节点：从 RDMA staging 取 token，经 LSA scatter 到目标 ranks/experts

combine：目标节点先把本节点 expert contributions 归约
         每个源 token 每个 node 只发一个 partial
源节点：再跨 node 归约并恢复原 token 顺序
```

NIC 字节数因此更接近“源 token × 命中的目标节点数”，而不是“源 token × 命中的 expert 数”。代价是更复杂的 routing maps、更大的共享 staging 和固定的节点内/节点间流水线。

### 9.2 Dispatch 前的 host/device 准备

HT forward 先将 sparse `[B,K]` weights 展开成 `[B,E]` dense probability。多节点且输入不是外部 NCCL window 时，普通 user token 会 D2D 拷到 group 创建时已注册的 staging，避免热路径临时注册。外部 window tensor 则保留其 window/offset，kernel 可直接对注册内存做 GIN 或 LSA 访问。

随后 runtime JIT 按以下维度生成专用 kernel：node 数、LSA size、layout、FWD/BWD、BF16/FP8 分支、H、block 数、stage/pipeline 参数。默认发射 16 blocks。

### 9.3 Dispatch warp-specialized pipeline

多节点 FLAT 每 block 有 6 个 warps，EM 多一个 PAD warp：

| Warp group | 数量 | 责任 |
|---|---:|---|
| inter-node/N2N | 2 | 扫描 outbound map；按远端 node 压缩 token/probability；GIN put + chunk tail signal |
| intra-node G2S | 2 | 本地直接取或等待远端 chunk；TMA global→shared，组成两个 pipeline |
| intra-node S2G | 2 | 查 S2D/rank mask；TMA shared→目标 LSA rank 的输出 slot |
| PAD（仅 EM） | 1 | 将 expert zone 的对齐 padding 清零 |

核心流水为：

1. N2N 以 64-token chunk 扫描 `attn_to_rdma_map`，只打包目标 node 真正需要的 token；一次 RDMA 可批量携带最多 4 个连续 token；
2. put 包含 token、forward dense probability，以及代码分支允许时的 scales；每个 chunk 用 GIN signal 发布 tail；
3. G2S 使用 12-stage、2-pipeline shared-memory FIFO。远端 chunk 先等待 tail，本地 chunk 直接从输入取；
4. S2G 读取 compact map，把 shared token 写到每个目标 LSA rank。FLAT 每目标 rank 一个 slot，EM 每目标 expert 一个 slot；
5. kernel 尾部执行 grid barrier，所有 block 完成远端/LSA 写入后更新 rank-0 completion flag；最后一个 block 重置 grid counter 并推进下一次调用的 expected generation。

expected counter 存在 device memory 中，并由 kernel 尾部递增，因此该状态更新会进入 CUDA Graph，replay 时不会一直等待旧 generation。

### 9.4 Dispatch 返回前的 copy 行为

HT kernel 默认写 group 的节点内 staging。若输出是外部 window，S2G 可直接写 user buffer；否则 host wrapper 再做 staging→user D2D copy。

这里存在重要的异步差异：

- 非 CUDA Graph capture：wrapper 将 `num_tokens_for_experts` D2H，随后 `cudaStreamSynchronize`，得到实际行数后只复制有效 rows；因此普通 HT dispatch **会阻塞 host**；
- capture 中：不能 D2H 同步，固定复制 `Mrecv` rows；输出必须按静态上界分配；
- 外部 window：token/scales 可省掉该 staging copy，但 forward dense→sparse weights/idx 转换仍在 stream 上执行。

即使 token 输出使用外部 window，非 capture 路径当前仍执行 actual-row 的 D2H readback 和 stream sync；外部注册只能消除 token staging/copy，不能消除这次 host 同步。

Forward 结束后，dense probability 被转回调用者布局：FLAT 输出 `[slot,K]` weights/local ids，EM 输出每 slot 一个 scalar weight。BWD dispatch 不携带 probability，只复用 handle 中的 routing maps。

### 9.5 Combine pipeline

HT combine 只接受 BF16 token。普通 expert 输出先拷入 IPC staging；外部 window 可直接被 LSA peer 读取。FWD 要求调用者已经按 layout 应用 routing weights；BWD 先把本地 sparse weight-gradient 通道展开成 dense expert probability。

进入实际搬运前，combine kernel 的 block 0 先在 LSA completion flag 上发布本 rank 已到达，所有 blocks 等到本 LSA team 的 generation 计数齐备后再读取 peer expert buffers。这一 head barrier 将“各 rank 的 expert 输出已按 stream 顺序准备好”发布给节点内消费者；它不替代应用在不同 stream 间建立的依赖。

多节点 combine 每 block 11 个 warps：

| Warp group | 数量 | 责任 |
|---|---:|---|
| intra-node G2S | 1 | 依照 S2D 从本节点各 rank 拉取 expert 输出 |
| intra-node reduction | 4 | FP32 累加本节点 contributions，按源 node 形成 partial |
| inter-node RDMA | 1 | 将 partial 按 64-token chunk 发回 token home node |
| inter-node G2S | 1 | 等待并加载本地/远端 node partials |
| inter-node reduction | 4 | 跨 node FP32 求和，写回原 token 顺序 |

单节点省掉第一阶段和 RDMA，使用 2 个 G2S warps + 4 个 reduction warps、两个数据 pipeline。多节点 reduction 每处理 8 个 token 就可向 RDMA warp 发布累计进度，而不是等待整个 64-token chunk，从而重叠节点内归约与网络发送。BWD 同一流水还归约 float probability，并最终按 cached `topk_idx` 恢复 `[B,K]`。

### 9.6 HT 的硬约束

- 官方 release 支持边界写明最多 8 nodes、每 node 8 ranks、总计 64 GPUs；代码未对所有越界情况做 host 早检；
- `Msend` 必须显式给出且不超过 8192，动态 token capacity 尚未实现；
- `Mrecv` 必须覆盖路由倾斜；EM 还要覆盖同 token 多 expert 复制和 padding；
- `experts_per_node` 必须为 4 的倍数，以满足 probability TMA 16-byte 对齐；
- token 的每行字节数必须为 16-byte 倍数；shared-memory 超限会在 launch 前失败；
- EM 每 rank 最多 64 个本地 experts；
- combine 仅 BF16；
- 源码虽然已有 HT FP8 分支，但 release notes 明确写“no FP8 support”，且当前初始化没有建立普通输出所需的 scale peer-pointer 数组，multi-node scale staging 也只按每 token 一个 float 分配/复制，而 kernel 按 `H/128` 个 float 读取。因此在这些问题修复并补齐测试前，应将 HT FP8 视为未完成、不可用能力。

---

## 10. 同步、并发与 CUDA Graph

| 边界 | 保证方式 | 使用者责任 |
|---|---|---|
| HT routing 全局一致 | base communicator 上的 `ncclAllGather` | 所有 rank 同序调用 `UpdateHandle`，B/K/容量契约一致 |
| LL payload ready | P2P release/acquire count 或 GIN put 后 signal | 不复用在途 output；正确调用 Complete |
| HT chunk ready | GIN tail signal + TMA mbarrier | 不绕过 handle metadata；容量不能溢出 |
| HT 节点内写完成 | rank-0 completion flag + expected generation | 同一 group 操作保持全 rank 同序 |
| HT grid 完成 | device grid counter，最后一个 block 清零并推进 generation | `max_num_sms` 与 launch 参数一致 |
| LL buffer 扩容/销毁 | device sync + NCCL barrier + window deregister | 所有 rank 无在途通信，既有 graph 重新 capture |

源码没有 group 级 mutex、operation queue 或每 handle 独立的数据面 buffer。LL handles 指向同一组 offset，HT handles 共享 IPC/GIN staging、completion counters 和 global routing bitmap。由资源所有权可以推断：**同一 EP group 是单一有序流水线，不应在多个 stream/handle 上并发 UpdateHandle、Dispatch 或 Combine**；至少要用 event/stream dependency 完全串行化。不同 EP group 是否可并发还取决于其 communicator、NIC context 和 SM 预算。

Graph 使用还要注意：

- 在 capture 前创建全部 LL handles，避免 AUTO buffer 后续扩容改变 base pointer；
- HT capture 走 `Mrecv` 固定长度 copy，不能依赖普通路径的 actual-row host readback；
- routing update 包含 NCCL AllGather 和 runtime JIT，最好在 capture 前完成并预热所需 variant；
- `send_only` 仅 LL 有效。HT 即使设置该字段，当前代码也不会报错，只是忽略它；不要依赖这种行为。

---

## 11. 容量与内存成本

容量必须按最坏 routing 而不是平均 routing 设计。最小检查思路为：

```text
FLAT Mrecv >= 此 rank 可能收到的唯一源 token 数
EM   Mrecv >= Σ(local expert e 的最大 token 数经 alignment 向上取整)
```

若只知道发送上界，可用保守界辅助规划：FLAT 最多为 `R×Msend`；EM 在 top-k experts 互异时，实际 rows 最多约为 `R×Msend×min(K,L)`，再加每个 expert zone 至多 `alignment-1` 个 padding。真实模型通常远低于该界，但库不会替调用者处理超额 token。

LL EXPERT_MAJOR 用户输出固定预留 `L × R × Msend` rows；RANK_MAJOR 固定预留 `R × Msend` rows。HT 更紧凑，但需要用户给出不会溢出的 `Mrecv`。

几项主要内存成本如下：

| 资源 | 近似规模/特点 |
|---|---|
| HT global bitmap | `R × Msend × ceil(E/8)` |
| HT dense probability | `Msend × E × 4 B` |
| HT intranode token staging | dispatch + combine 各 `Mrecv × max_token_bytes` |
| HT intranode probability staging | dispatch + combine 各 `Mrecv × experts_per_node × 4 B` |
| HT multi-node partial buffers | 若干 `8192 × (N-1) × token/probability bytes`，按编译期上限而非实际 `Msend` |
| LL dispatch header | EM `align16(4+2K)`；RM `align16(4+8K)`，再加 token payload |
| LL group buffer | 两个 ping-pong slot，各取 dispatch 与 combine send+recv 的较大值，再加信号区 |
| LL handle source map | 约 `R × (1 + Msend × (K+1)) × 4 B` |

HT 的层次化方案用内存换网络效率；特别是 multi-node buffer 的若干区域固定按 8192-token stride 分配，即使运行时 `Msend` 很小也不会同比缩小。大 `E` 又会放大 dense probability 和 bitmap。上线前应实测 group/handle 峰值，而不能只统计用户 token tensor。

---

## 12. 超时、mask 与恢复语义

active mask 只在 LL 实现。mask 是 device `int[R]`：1 表示 active，0 表示 masked。

- 未开启 mask：等待超过 timeout 后 device trap；
- 开启 mask：超时 peer 被置 0，mapped pinned host error flag 置 1，后续发送和归约跳过该 peer；输出是缺少失败 rank contribution 的**降级部分结果**，不是数学上完整的 MoE 结果；
- 各 rank 独立观察超时，mask 可能不一致，框架必须查询、对齐并执行 expert rebalance 或其他恢复策略；
- `ncclEpMaskQuery` 的 `int*` 实际必须是 device pointer，因为实现执行 D2D copy；
- `ncclEpMaskUpdate` 的输入必须是 host pointer，因为实现执行 H2D copy；
- `ncclEpMaskClean` 是 collective：mask-aware barrier → 清两套 LL count/flag buffer → barrier → 全部置 active，并同步传入 stream；至少要先创建一个 LL handle，使主 RDMA buffer 已分配；
- `ncclEpErrorClear` 只清 host error flag，不清 buffer、不恢复 mask。

HT 没有 mask/recovery 数据路径。任一 rank 缺席、路由不一致或 signal 不前进，当前语义不是弹性继续运行。

---

## 13. Runtime JIT、构建与可观测性

LL kernel 随库静态编译；HT 的 metadata scan、dispatch 和 combine 必须在运行时调用外部 NVCC 生成 cubin，没有可用的静态 fallback。任一 JIT compile/load/launch 失败都会打印 fatal 信息并 `abort()`；runtime 日志中的“using static path”只是遗留文案，与调用方行为不符。

JIT 的关键行为：

- NVCC 查找顺序：`NCCL_EP_JIT_NVCC`、`NVCC`、`CUDA_HOME/CUDA_PATH/bin/nvcc`、PATH；
- 默认 cache 为 `/tmp/nccl_ep/jit`，可由 `NCCL_EP_JIT_CACHE_DIR` 改写；
- cache 保存 `kernel.cu`、`kernel.cubin`、`metadata.json`、`compile.log` 和失败 marker，并用文件锁避免多进程重复编译；
- `NCCL_EP_JIT_LOG=1` 输出细节；`NVCC_ARCH_FLAGS`、`NVCC_EXTRA_FLAGS` 可覆盖编译参数；
- 首次遇到新的 SM、H、layout、node/LSA size、方向或 dtype variant 会阻塞编译。部署镜像必须包含 NVCC、JIT headers 和可写 cache，生产环境应预热所有 shape。

官方 README 的支持前提是 CUDA 13+、NCCL 2.29+，测试平台为 Hopper/Blackwell；Makefile 本身允许 CUDA 11.8+ 编译 sm90。两者不等价：能编译不代表位于 release 支持矩阵，应以目标 release 的 README/RELEASE 为准。

构建入口是：

```bash
# 仅适用于源码树已经包含 contrib/nccl_ep 的版本
# 先构建 NCCL，再构建独立 EP 库
make -C nccl src.build
make -C nccl/contrib/nccl_ep

# MPI test/bench（源码合入本地树后）
make -C nccl/contrib/nccl_ep MPI=1 ep_test ep_bench
```

`WARP_TIMING=1` 会启用 HT warp-level instrumentation，但每次 launch 额外分配、拷贝并释放 timing buffer，会改变性能。`ep_bench` 带 NVTX，并在 CUPTI headers 可用时记录 kernel timing。

上游测试目前覆盖 tensor descriptor、生命周期、handle maps、HT backward、stale routing tail，以及 FLAT/EM/LL output layout；release notes 同时标记 limited QA。明显缺口包括多节点故障恢复、并发 handles/streams、JIT 故障、容量边界、完整 FP8 和超出官方拓扑范围的行为。

---

## 14. 适用场景与选择建议

| 场景 | 建议 | 原因 |
|---|---|---|
| autoregressive decode，单 rank token 很少 | LL / EXPERT_MAJOR 起步 | 最短 direct/GIN 路径；expert-ready layout；无需全局 bitmap AllGather |
| decode，但同 rank 常命中多个 experts且返回流量敏感 | 评估 LL / RANK_MAJOR | 同 rank 预归约后每 expert rank 只返回一个 contribution |
| 训练或大 prefill | HT | 按 node 去重、层次化归约、TMA/warp pipeline 更能摊薄 metadata 与同步成本 |
| expert kernel 要求连续且对齐的 per-expert batch | HT / EXPERT_MAJOR | 直接生成 expert zones，可配置 2 的幂 padding |
| 框架已有高效本地 expert regroup/pre-reduce | HT / FLAT | token slot 更少，避免同 rank 多 expert 的 token 复制 |
| 单节点 NVLink/NVSwitch | LL/HT 都应基准测试 | LL 延迟低；HT 仍可利用 TMA pipeline，但其预处理和 staging 成本未必值得 |
| 动态极强、无法给接收上界的 routing | 当前 EP 不适合 | 动态 capacity 尚未实现，溢出会 trap |
| 异构 expert placement、每 rank expert 数不同 | 当前 EP 不适合 | 强制 `E % R == 0` 和固定连续 owner 关系 |
| 没有 Device API/GIN、没有运行时 NVCC | 多节点 EP 不适合 | LL 多域和 HT 多节点依赖 GIN；HT 还强依赖 JIT |

选择不能只看 token 数，还要同时测量：路由命中的 rank/node 数、top-k 去重率、expert imbalance、H、局部预归约成本、Mrecv 浪费、NIC/LSA 拓扑、SM 争用，以及 HT 普通输出路径的 host sync/copy。

---

## 15. 设计优势

1. **MoE 语义下沉。** rank/node 去重、compact slot、expert zone、反向映射和 combine reduction 都在库内，不再只是发送框架预打包的字节。
2. **GPU 主导数据面。** LSA direct store 和 GIN put/signal 由 kernel 发起，避开普通 NET proxy 的 CPU critical path。
3. **拓扑分层。** HT 用 LSA + rail team 把节点内复制/归约与节点间 RDMA 分开，显著减少重复 NIC payload。
4. **预注册和长期复用。** group 预建 window、Device Comm、signals 和 staging，避免每步注册；handle 缓存逆映射，combine 不必重新解释 routing。
5. **计算通信流水。** LL 可拆 send/receive；HT 用 warp specialization、TMA stage、chunk signal 和 streaming reduction 重叠多阶段工作。
6. **layout 与训练方向较完整。** FLAT/EXPERT_MAJOR/RANK_MAJOR 将不同框架责任显式化，HT 还复用同一 routing 支持 forward/backward。
7. **Graph-aware generation。** HT expected counters 在 kernel 尾部推进，避免 graph replay 使用陈旧 flag；capture 路径避免 host readback。

---

## 16. 不足、风险与实现债务

### 16.1 已由源码直接确认

- 仍是 `contrib` 下的 0.1.0 experimental library，且不在本仓库 NCCL 2.29.7 源码中；
- API 注释与实现存在漂移：HT `num_topk=-1`、动态 token、FP8、`Complete` stream、JIT fallback 等表述不能直接相信；
- 错误处理混用 `ncclResult_t`、C++ exception、`assert`、`exit`、`abort` 和 device trap。若从 C ABI 触发未捕获异常或 fatal 路径，进程可能直接终止；
- LL 真实 K 上限为 9、hidden 只支持七个模板值，均比公共常量/通用 tensor API 更窄；
- `topk_idx` 没有完整的 host 范围/去重校验，非法 expert id 可在 HT bitmap kernel 中造成越界写，而不只是返回参数错误；
- HT 的静态容量、8192 stride 和 dense probability 带来较高显存成本；没有 drop/capacity-factor 策略；
- 普通 HT dispatch 为获得实际 row 数会同步 host，削弱纯异步 enqueue 语义；
- `num_channels` 当前无实际作用；`max_num_sms/num_qp` 只是固定默认，没有测量式 tuning；
- HT JIT 要求生产节点存在 NVCC、headers、可写文件系统；首次编译延迟和 fatal failure 增加部署复杂度；
- active-mask 只覆盖 LL，而且降级结果、mask 一致性和 expert rebalance 都交给上层；
- group destroy 与 LL buffer 扩容包含全设备同步或 collective barrier，生命周期操作成本高；
- API 没有强制 RDMA-only 或显式 fabric 选择开关，实际 direct/GIN path 由 NCCL team、window 可达性和 GIN 能力决定；
- RELEASE 声明最多 8×8 GPUs、limited QA、HT 性能仍在调优。

### 16.2 从共享状态推导出的工程风险

- 多 handle 并发会争用 group routing bitmap、staging、signals、generation counters 和 LL slots；当前没有隔离或序列化机制；
- `UpdateHandle` 和数据面跨 stream 使用时，库只依赖调用者提供正确顺序，缺少事件或版本号验证；
- timeout 后不同 rank 的 mask 可分叉，继续运行可能得到彼此不一致的部分结果；
- hostname 识别、LSA team、rail team 和 rank 编排必须一致，非对称/容器化 hostname 环境需要额外验证；
- 代码对不支持的 shape 常在较深处 assert/trap，生产框架必须在 API 外建立更严格的预检查和故障隔离。

---

## 17. 接入与排查清单

### 初始化前

- 确认使用的 NCCL 源码确实包含目标版本 `contrib/nccl_ep`，并记录 EP/NCCL/CUDA commit；
- 检查 `deviceApiSupport`、LSA/rail team 大小、GIN 类型和官方 8×8 支持边界；
- 保证 `E % R == 0`、`K>0`，按最坏路由估算 `Mrecv`；
- 验证每个 `topk_idx` 都是 `-1` 或 `[0,E)` 内的互异值；不要把 GPU kernel 当作输入校验器；
- LL 提前检查 `K<=9`、hidden 模板值、`Msend×L % 4 == 0`；
- 多 LSA domain 的 LL 至少配置 4 个可用 GIN contexts，并核对 AUTO 请求值是否超过 plugin 能力；超过 4 个目前不会提高 LL kernel 的 context 并行度；
- HT 检查 TMA 对齐、EM `L<=64`、显存预算和 runtime NVCC/cache；
- 在 graph capture 前创建所有 handles，并预热 scan/dispatch/combine JIT variants。

### 每一步

- 所有 rank 以相同顺序执行 routing update、dispatch 和 combine；
- 同一 group 不并发操作；staged LL 必须一一配对 `Complete`；
- 正确区分 layout 中“谁应用权重、谁做本地预归约”；
- 只读取 metadata 给出的有效 rows/zones，不处理 padding；
- 监控 actual receive count、expert imbalance、staging copy、host sync、GIN wait 和 kernel SM 占用。

### 故障时

- LL 开 mask 时同时查询所有 rank mask，先达成一致再决定降级、rebalance 或重建 communicator；
- `ErrorClear` 不等于恢复，重新接纳延迟 rank 要执行 collective `MaskClean`；rank 替换必须重建 communicator 和 EP group；
- JIT 失败检查 cache 下的 `compile.log/compile.failed`，修复后清理对应 variant 的失败 marker，而不是期待静态 fallback；
- device trap/容量溢出后不要复用旧 group 状态，应视 CUDA context 状态决定重建进程或 communicator。

---

## 18. 源码审阅索引

以下链接固定到本文审阅提交，避免 `master` 后续变化使结论失去对应关系：

- [NCCL EP README](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/README.md)
- [公开 API：nccl_ep.h](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/include/nccl_ep.h)
- [枚举与 layout：ep_enums.h](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/include/ep_enums.h)
- [Host runtime：nccl_ep.cc](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/nccl_ep.cc)
- [LL kernel：low_latency.cu](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/device/low_latency.cu)
- [HT kernel：hybrid_ep.cuh](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/device/hybrid_ep.cuh)
- [HT adapter：hybridep_adapter.cu](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/device/hybridep_adapter.cu)
- [Metadata scan：scan_kernel.cuh](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/device/scan_kernel.cuh)
- [Runtime JIT](https://github.com/NVIDIA/nccl/tree/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/device/jit)
- [Release limitations](https://github.com/NVIDIA/nccl/blob/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/RELEASE.md)
- [Tests](https://github.com/NVIDIA/nccl/tree/5067397c2676d5aed50042fc39e5c8ee96eb0027/contrib/nccl_ep/tests)

本文结论来自上述源码的静态审阅；当前工作环境未具备对应多节点 Hopper/Blackwell 拓扑，未执行 EP GPU 正确性或性能测试。因此，支持范围、性能拐点和故障恢复仍应在目标硬件、目标 NCCL plugin 与真实 MoE routing 分布上验证。
