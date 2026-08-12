# NCCL EP：从 MoE 入门到源码实现

本文基于当前仓库 NCCL 2.31.2-1 中的 `contrib/nccl_ep/` 源码重新审核。这里的 **EP** 是
**Expert Parallelism（专家并行）**，不是网络里的 endpoint。NCCL EP 当前版本为 0.1.0，
是建立在 NCCL Device API 之上的独立扩展库，不属于顶层 `make src.build` 的核心产物。

仓内 README 已声明 NCCL EP 的后续开发迁移到 `NVIDIA/nccl-extensions`；本章描述的是当前
NCCL 仓库快照中的实现。要提交新 issue/PR 或判断更新版本行为，应同时检查新仓库，不能默认
这里仍是唯一事实来源。

## 1. 先用一句话理解它

MoE（Mixture of Experts，混合专家）层不会让每个 token 经过所有专家，而是由 router 为它挑选
top-k 个专家。专家权重又分散在多张 GPU 上，因此一次 MoE 前向包含：

```text
原始 token
   │ router 产生 topk_idx / topk_weights
   ▼
dispatch：把 token 送到专家所在 GPU
   ▼
各 GPU 执行本地 expert MLP
   ▼
combine：把专家结果送回 token 出生的 GPU，按权重求和并恢复原顺序
```

NCCL EP 专门实现其中的 `dispatch` 和 `combine`。它不是在普通 All-to-All 外面简单包一层，
而是把路由解析、跨 GPU 搬运、接收槽位分配、元数据传递和返回归并一起设计，并让 GPU
通过 LSA/GIN 主动通信，尽量不让 CPU 参与热路径。

## 2. 为什么通用 NCCL collective 不够顺手

假设有 4 个 rank、8 个专家，每个 rank 保存 2 个连续编号的专家。token A 的 router 选择专家
1 和 6，它就要从出生 rank 分别去 rank 0 和 rank 3；计算完成后，两份结果再回到出生 rank
相加。不同 token 的目标、每个目标收到的数量都会变化，这不是固定 split 的规则 All-to-All。

直接用通用通信原语，框架通常还要自行完成：统计每个 peer 的 token 数、交换计数、打包、
All-to-All、解包、按专家重排、保存反向映射、返回和加权归并。NCCL EP 的设计意图是：

- 让 API 直接理解 `topk_idx`，减少框架重复实现路由通信的工作。
- 将同一目的 rank 的重复路由合并，避免一个 token 在同一 GPU 有多个目标专家时重复传输。
- 同一套 API 同时覆盖小 batch 的低延迟推理和大 batch 的高吞吐训练。
- 复用 NCCL 已发现的 NVLink/网络拓扑、内存窗口、Device API 和网络插件。
- 把通信协议做进持久 GPU 数据结构和专用 kernel，避免每轮注册网络内存或由 CPU 推进。

典型使用场景是稀疏 MoE LLM 的 decode、prefill 和训练；它不负责 router、expert MLP、负载均衡
loss，也不替代数据并行或张量并行。

## 3. 阅读源码前必须认识的名词

- **rank**：NCCL communicator 中的一个参与者，通常对应一个进程和一张 GPU。
- **home/source rank**：token 原本所在的 rank。
- **expert rank**：保存目标专家参数、执行 expert MLP 的 rank。
- **top-k**：一个 token 选择的专家数；`topk_idx[B,K]` 是全局专家编号，
  `topk_weights[B,K]` 是 router 权重。
- **local expert**：本 rank 上的专家。当前代码按连续编号平均切分，
  `num_local_experts = num_experts / nRanks`，因此总专家数必须能被 rank 数整除。
- **LSA team**：Load-Store Accessible team，GPU 之间可通过 NVLink 等直接 load/store 的域。
- **rail team**：跨节点时同一 rail 上的 rank 集合，HT 用它做分层通信。
- **GIN**：GPU-Initiated Networking。kernel 内通过 `put`/`signal` 发起 RDMA，而不是交给 CPU proxy。
- **TMA**：Hopper 的 Tensor Memory Accelerator，HT kernel 用它异步搬运规整 tensor tile。

## 4. 代码地图

| 路径 | 作用 |
|---|---|
| `contrib/nccl_ep/include/nccl_ep.h` | C API、配置、tensor 描述符和 ABI 约定 |
| `contrib/nccl_ep/include/ep_enums.h` | LL/HT、layout、前向/反向枚举及语义 |
| `contrib/nccl_ep/nccl_ep.cc` | 参数检查、group/handle 生命周期、内存、API 到 kernel 的适配 |
| `contrib/nccl_ep/include/common.hpp` | 公共常量、LL RDMA buffer 布局、内部结构 |
| `contrib/nccl_ep/nccl_ep_env.{h,cc}` | 每个 group 的环境变量解析与诊断输出 |
| `contrib/nccl_ep/device/ll_ep.cuh` | LL dispatch/combine/clean 设备端模板实现 |
| `contrib/nccl_ep/device/ll_ep_adapter.{cuh,cu}` | LL 参数打包、SM/warp 选择和 JIT 启动 |
| `contrib/nccl_ep/device/device_primitives.cuh` | LSA、GIN、等待、barrier 等设备原语 |
| `contrib/nccl_ep/device/scan_kernel.cuh` | HT 路由 bitmap 扫描和槽位映射预处理 |
| `contrib/nccl_ep/device/hybridep_adapter.cu` | HT 参数适配、JIT 变体选择与启动 |
| `contrib/nccl_ep/device/hybrid_ep.cuh` | HT warp-specialized dispatch/combine 主体 |
| `contrib/nccl_ep/device/jit/` | LL/HT CUDA 源码生成、编译缓存和 driver module 加载 |
| `contrib/nccl_ep/ep_test.cu`、`ep_bench.cu` | MPI 正确性示例和性能测试 |
| `contrib/nccl_ep/tests/` | 不依赖 MPI 的多 GPU gtest |

## 5. API 对象与完整生命周期

### 5.1 三类对象

**`ncclEpTensor_t`** 只是描述符，不拥有 tensor 数据。它记录维数、类型、shape，以及普通 device
pointer，或 NCCL window handle 加 offset。栈上对象应以 `NCCL_EP_TENSOR_INIT` 初始化；
`sizes` 数组在调用期间必须仍有效。`ncclEpTensorAlloc` 只分配描述符并复制 shape，不分配 GPU
数据，最后用 `ncclEpTensorDestroy` 释放描述符。

**`ncclEpGroup_t`** 是长生命周期资源，绑定一个 NCCL communicator 和一种算法。它保存 rank/
拓扑、专家切分、容量上限、NCCL device communicator、对称窗口、GIN 注册区、IPC 指针、同步
flag 和 workspace。自定义 allocator 必须同时提供 alloc/free，否则创建失败。

**`ncclEpHandle_t`** 对应一份可复用的路由状态。它保存 `layout`、token/top-k 数、`topk_idx`
的永久描述符副本，以及 dispatch 后 combine 所需的反向映射。路由变化时调用
`ncclEpUpdateHandle`，无需重建 group。当前 LL/HT 都接受 int32 或 int64 的 `topk_idx`，并按
调用者原始宽度缓存；这能减少大型路由表的显存和带宽，但调用者要保证 expert id 不溢出 int32。

“描述符副本”不等于复制用户数据：shape 数组由库深拷贝，但 `.data`/window 仍指向调用者存储。
HT UpdateHandle 会把路由值另存进 handle memory；LL dispatch/combine 仍直接读取该用户路由
buffer。因此 LL 必须让 `topk_idx` backing storage 至少活到最后一次 combine 完成。

所有跨 API 的公开 struct 都以 `size` 和 `magic` 开头，group config 还带 API `version`。
务必使用对应的 `NCCL_EP_*_INIT`；这既是初始化，也是 ABI 检查。大多数 struct 当前要求 size
严格相等；为新增 `recv_topk_idx_kind`，只有 `ncclEpLayoutInfo_t` 特殊接受从旧版最小 size 到
当前 size 的范围，缺失字段按 AUTO 处理。零长度 tensor 也有特殊放宽：任一维为 0 时允许没有
data/window binding，但各 rank 仍需参与 collective。

### 5.2 推荐调用顺序

```text
ncclCommInitRank
  └─ ncclEpCreateGroup              所有 rank 集体调用，创建长期资源
       └─ ncclEpCreateHandle        = InitHandle + UpdateHandle
            ├─ ncclEpDispatch       使用 handle 中缓存的 topk_idx
            ├─ 本地 expert MLP      由上层框架执行
            └─ ncclEpCombine
       ├─ ncclEpUpdateHandle        下一批路由改变时更新并复用
       └─ ncclEpHandleDestroy
  └─ ncclEpGroupDestroy
ncclCommDestroy
```

`ncclEpInitHandle` 与 `ncclEpUpdateHandle` 也可拆开使用，便于由框架提供 handle memory；但当前
LL 和 HT 的 Init 路径都要求正的 `num_topk`。虽然头文件部分注释仍写 HT 可传 `-1`，实现会
assert；对初学者，直接用 `ncclEpCreateHandle` 最安全。
group、handle update（HT 会 AllGather 路由）和通信操作都应由各 rank 以一致顺序参与。

API 将工作排到传入 CUDA stream，tensor 内存由调用者预分配。若创建/更新 handle 与通信使用
不同 stream，调用者负责建立依赖。销毁 group 会 `cudaDeviceSynchronize`，不是热路径操作。

## 6. 四种“算法 × 布局”组合

令 `B` 为本 rank token 数，`H` 为 hidden，`K` 为 top-k，`R` 为 rank 数，`L` 为本地专家数，
`M` 为每个 rank 最大 dispatch token 数。

| 算法 | layout | dispatch token 输出 | 谁处理权重 |
|---|---|---|---|
| LL | Expert Major | `[L, R*M, H]` | combine 在源 rank 按原始权重加权 |
| LL | Rank Major | `[R, M, H]`，另有 `[R,M,K]` 权重/专家号 | 调用者先在专家 rank 按本地专家加权归并 |
| HT | Flat | `[max_recv_slots,H]`，另有 `[slots,K]` 权重/本地专家号 | 调用者路由本地专家并预归并 |
| HT | Expert Major | `[max_recv_slots,H]`，按专家分区；权重 `[slots]` | 调用者逐槽应用权重 |

LL Expert Major 的第二维必须是 `R*M`。当前 `nccl_ep.h` 的 dispatch 参数注释有一处写成
`[L,M,H]`，但 `ep_enums.h`、实现断言、README 表格和示例均以 `[L,R*M,H]` 为准。

布局的核心取舍是“是否复制”：Expert Major 为每个命中的本地专家生成一个槽位，expert MLP
最容易直接消费，但一个 token 命中同 rank 多个专家时会复制；Rank Major/Flat 尽量每个目标
rank 只保留一份 token，节省带宽和空间，但调用者要依据专家号在本地再分发并归并。

LL Rank Major 和 HT Flat 会输出 `recv_topk_idx`。新增的
`ncclEpLayoutInfo_t::recv_topk_idx_kind` 可明确指定编号语义：

- `NCCL_EP_EXPERT_ID_LOCAL`：写本 rank 内的专家号 `[0,L)`，不属于本 rank 的位置写 `-1`。
- `NCCL_EP_EXPERT_ID_GLOBAL`：保留 router 使用的全局专家号，空位置仍是 `-1`。
- `AUTO`：当前解析为 LOCAL，但未来允许改变。框架若依赖稳定语义，应显式设置 LOCAL/GLOBAL。

注意输入路由可为 int32/int64，但输出类型并不统一：当前 LL Rank Major 的 `recv_topk_idx` 要求
int32，HT Flat 要求 int64；“编号范围”和“tensor 元素宽度”是两个独立概念。

## 7. Group 创建到底做了什么

`ncclEpCreateGroup` 是理解全局资源的入口：

1. 校验算法、容量和 Device API 支持，读取 communicator 的 rank、GPU 和 NCCL team。
2. 用 hostname 的 host AllGather 统计节点数，计算 node/rank 位置；当前实现隐含各节点 rank
   数量均衡、rank 排列规整的假设。
3. 一次性读取 `nccl_ep_env`，解析通信/prolog/preprocess 三类 SM、QP、chunk 和超时；
   `NCCL_EP_TIMEOUT_MS` 的优先级高于 config。
4. 分配公共 workspace。
5. LL 创建一个 full-world device communicator；跨 LSA 时要求 GIN full connections。RDMA
   buffer 可显式立即分配，也可等首个 handle 根据 layout/top-k 延迟分配。
6. HT 根据 `ncclTeamLsa` 和 `ncclTeamRail` 建立二维拓扑，要求
   `lsa_team_size * rail_team_size == nRanks`；随后一次性分配并注册节点内对称窗口、跨节点
   GIN buffer、completion/guard flags 和全局路由 bitmap。即使只有一个 LSA 域，也创建最小
   device communicator，供新的 LSA 跨轮 guard barrier 使用。

这里的原则是“昂贵动作前置”：尤其 HT 不会在每轮 dispatch 临时注册 GIN 内存，因为源码注释
测得注册可能是几十毫秒量级。普通用户 tensor 会异步复制到预注册 staging；若 tensor 直接绑定
已注册的 NCCL window，则可以走 zero-copy 地址。

## 8. LL：低延迟路径详解

### 8.1 适用场景与总体策略

LL 面向 decode、小 micro-batch 等延迟敏感场景。它不先做全局统计或分层聚合，而是让每个
token 直接发往目标专家 rank：同一 LSA 内写 peer pointer，跨 LSA 用 GIN RDMA `put`。少量
消息的路径更短，代价是大 batch 下远端操作数较多、带宽聚合能力不如 HT。

### 8.2 双缓冲和内存布局

`LowLatencyLayout` 把一块注册 RDMA buffer 划成两个奇偶 buffer。每次调用取得当前 buffer，
同时切换 `buffer_idx`，并清理下一块的计数/flag。dispatch 与 combine 不重叠使用的发送、接收
区域会复用空间，降低容量。handle 中保存的是相对 offset 而不是裸指针，所以 AUTO 模式扩容
后旧 handle 仍可在新 base 上解析。

每份消息包含 token id、目标专家信息和 payload；Rank Major 还带权重。发送端先把一个 token
的 top-k 按目标 rank 去重：若专家 4、5 都在 rank 2，token 数据只发一次，header 记录两项路由。

### 8.3 Dispatch 的设备端流程

```text
每个源 token
  ├─读取 topk_idx，算出 expert_owner_rank
  ├─按目标 rank 去重并申请发送槽
  ├─同节点：直接写目标 peer buffer
  └─跨节点：GIN put(header + payload [+ FP8 scales])
                         │
                         └─最后写 count / signal
目标 rank 等待 count/signal
  ├─Expert Major：按 local expert 原子申请槽，复制到专家分区
  ├─Rank Major：按 source rank 申请槽，输出权重和本地 expert id
  └─记录 source token/rank 和 dispatch layout，供 combine 反向寻址
```

同节点消息把 header 区和 payload 区分开，跨节点消息则交错存为 header/payload/scales。无论哪种
路径，数据先写、completion 后写；GIN 对同 endpoint/QP 的 put-before-signal 顺序保证接收方
看到 signal 时数据已可见。等待不是 CPU 轮询，而是 kernel 内查看 peer counter 或 GIN signal。

最新 LL 的普通 token 通路支持 BF16、FP16、FP32；dispatch 还支持两种 FP8：

- **INTERN**：普通输入配 `outputs.scales`，kernel 在通信时量化 token 并输出 FP32 scale；当前
  JIT 的量化 wire specialization 固定按 BF16 组织，实际使用应以 BF16 输入为安全约束。
- **EXTERN**：输入本身是 E4M3/E5M2，并同时提供 `inputs.scales` 和 `outputs.scales`，只搬运
  已量化 token 与 scale。scale 可为 FP32 或 UE8M0（uint8），块大小由 shape 推导。

hidden 仍需满足 128/向量化约束。combine 不接受 FP8，但普通 BF16、FP16、FP32 均可返回归并；
输出 dtype 必须与 combine 输入一致。

### 8.4 Expert Major combine

dispatch 已把每个接收槽的来源记录到 `expert_recv_source_indices`，并在
`expert_dispatch_layout` 保存返回位置。expert MLP 原地或另存产生 `[L,R*M,H]` 输出后，combine：

1. 专家 rank 按保存的来源把每个 expert 结果发回 home rank。
2. home rank 等待各 top-k 返回槽完成。
3. 用 TMA 将多个向量装入 shared memory，以 FP32 累加。
4. `combine_outputs.topk_weights[B,K]` 提供原始 router 权重，kernel 边归并边乘权重。
5. 结果转换回调用者的 BF16/FP16/FP32 dtype，按原始 token id 写回 `[B,H]`。

### 8.5 Rank Major combine

dispatch 输出每个来源 rank 的 token、local expert id 和权重。上层必须自行运行命中的本地
experts，并先把同一 token 在本 rank 上的多专家结果做加权和，产生每槽一个向量。combine
只把这个“每专家 rank 的部分和”发回 home rank；home rank 再把不同 rank 的部分和相加，
此时不再乘权重。漏掉上层预归并会得到数值错误，而不是 API 报错。

### 8.6 `send_only` 如何实现重叠

LL kernel 有 `LOW_LATENCY_SEND_PHASE` 和 `LOW_LATENCY_RECV_PHASE` 两个位。正常调用一次启动
两阶段；`send_only=1` 只启动发送阶段，并把包含本次所有指针/参数的 closure 存进 handle。
稍后 `ncclEpComplete` 用同一个 closure 启动接收阶段。这样多个 micro-batch 可在网络传输时让出
SM 做计算。

一个 handle 同时只能保存一个待完成 closure；在 `Complete` 前再次对同一 handle 发起
`send_only` 会覆盖它。AUTO RDMA buffer 扩容也会丢掉未完成阶段的数据，所以必须先 drain。

### 8.7 LL 的 JIT 与 zero-copy

LL 已不再从 `low_latency.cu` 预编译所有 kernel。host 路径在 `ll_ep_adapter.cu` 计算 SM/warp、
打包一个固定参数 struct，再按 hidden、layout、量化模式、是否纯 NVLink、top-k 索引宽度和 token
dtype 生成 JIT 变体。dispatch/combine 使用 cooperative launch；block 数为偶数时请求 2-SM
cluster，clean kernel 也通过同一 JIT runtime 启动。因此 LL 与 HT 都存在首次编译冷启动。

`zero_copy=ON` 对 LL 是严格且很窄的契约：目前只支持纯 LSA/NVLink、Rank Major、非 FP8，且
`dispatch_outputs.tokens` 必须是 NCCL window-backed。发送方会直接写对端用户输出，接收方跳过
staging copy。LL combine 仍将内部 `zeroCopy` 参数固定为 false；group 的公开 zero-copy 开关
主要改变 dispatch 输出路径。

### 8.8 LL active mask 容错

设置 `enable_mask` 后，每个 rank 有一个 device mask（1 活跃、0 屏蔽）和 mapped pinned host
错误标志。GPU 等待远端超时时，不再直接 trap，而是原子屏蔽该 rank、设置异步错误并继续跳过。
应用可用 `ncclEpMaskQuery/Update` 管理 mask，用 `ncclEpGetAsyncError` 轮询，再以
`ncclEpMaskClean` 做跨 rank barrier、清计数和恢复协议状态；`ncclEpErrorClear` 单独清错误。

这只是通信层的“继续前进”机制，不会自动重建 communicator、替换 rank 或保证各 rank 做出
一致故障判断。框架必须协调 mask；真正替换进程/GPU 后应创建新 communicator 和 group。

## 9. HT：高吞吐路径详解

### 9.1 为什么要分层

HT 面向训练和 prefill 的大量 token。逐 token 对全世界发 RDMA 很难饱和链路，因此先利用
NVLink 在节点内聚合/散布，再由每个 rail rank 负责跨节点大块传输，形成：

```text
源 GPU ──节点内聚合──> 本节点 rail GPU
                         │ GIN RDMA
                         ▼
目标节点 rail GPU ──节点内散布──> expert GPU
```

`lsa_rank` 表示节点内位置，`rdma_rank` 表示节点/rail 位置。这一二维分解也是 group 创建时检查
team 大小乘积等于 world size 的原因。

### 9.2 路由预处理：UpdateHandle 是 HT 的关键

HT 不希望主通信 kernel 边走边解析稀疏 top-k，因此 `ncclEpUpdateHandle` 先把路由编译成紧凑
映射：

1. 将本 rank 的 `topk_idx[B,K]` 转为 `[max_tokens,ceil(E/8)]` bit-packed expert bitmap；
   未使用的尾行也清零，防止复用 handle 时残留旧路由。
2. 用 NCCL `ncclAllGather` 收集所有 rank 的 bitmap。因此 UpdateHandle 是 collective。
3. metadata scan kernel 计算计数、前缀和和目标槽位，生成：
   - `token_rank_mask`：一个 token 要去哪些 rank；
   - `sparse_to_dense_map`：稀疏路由项到接收槽的映射；
   - `rdma_to_attn_map`：远端到达项如何放回节点内 attention/token 顺序；
   - `attn_to_rdma_map`：本地项应发往哪个远端位置；
   - `local_expert_routing_map`：接收槽命中了哪些本地专家；
   - 每专家实际 count、offset 和总接收数。
4. 超过 `max_recv_tokens_per_rank` 时 device assert/trap，避免静默越界。

Flat 的槽按“来源 token 到目标 rank”去重；Expert Major 的槽按 `(来源 token, 本地 expert)`
展开，并可通过 `dispatch_output_per_expert_alignment` 把每个专家分区补齐到 2 的幂次对齐。

### 9.3 Expert Major 的三种内部实现

对用户而言 HT Expert Major shape 不变，但最新代码会在 group 创建时选择内部 fan-out 策略：

1. **Local Permute（默认 staging 路径）**：主 HT kernel 先按 Flat/Rank-Major 方式只传一份 token，
   随后本地 permute kernel 把它复制进各专家分区；combine 前用 reduce/permute 收回 Flat staging。
2. **NVLink Dup**：发送侧 S2G 直接通过 NVLink 写每个远端 expert slot。`zero_copy=ON` 且 LSA
   大于 1 时自动选择，也可用 `NCCL_EP_HT_EM_NVLINK_DUP=1` 强制。
3. **Local Dup**：主 kernel 只写 primary slot，目标 GPU 再运行 `local_dup` 填 secondary slots；
   combine 前 `local_reduce` 把 secondary 累加回 primary。单 GPU LSA 的 zero-copy 会自动选择，
   也可用 `NCCL_EP_HT_EM_LOCAL_DUP=1` 强制。

两个环境开关互斥。Local Permute 已支持 BF16/FP16/FP32，但不支持 FP8；Local Dup 的辅助 JIT
目前也把 token 类型固定为 16 位，FP8 会被拒绝。FP8 Expert Major 若要走通，需要 NVLink Dup。

### 9.4 Warp-specialized dispatch

HT 不是一个 warp 做完所有工作，而是在同一个 cooperative kernel 中按角色分工：

- **N2N warp**：按 chunk 打包跨节点 token/概率/scale，执行 GIN put，再发 signal。
- **G2S warp group**：等待节点内或 RDMA 数据，通过 TMA 从 global 搬到 shared。
- **S2G warp group**：查询预处理映射，通过 TMA 把 shared tile 散布到目标 GPU 的输出槽。
- **PAD warp**：Expert Major 时把对齐产生的 padding 槽清零。

末尾用 grid barrier 和跨 rank completion flag 收口。flag 使用单调递增的 expected value，而不是
每次把整片内存清零，这使重复执行和 CUDA Graph replay 更便宜。GIN 同样遵循 data put 在前、
signal 在后的同 QP 排序。新实现还默认启用跨轮 WAR guard：下一轮覆写共享 LSA/RDMA staging
前先确认上一轮消费者已结束，防止相邻 dispatch/combine 互相破坏。只有能由框架严格保证无重叠
时，才应以 `NCCL_EP_DISABLE_GUARD=1` 关闭它。

HT dispatch 的普通通路支持 BF16/FP16/FP32，也支持带 scale 的 E4M3/E5M2 外部 FP8 输入。FP8
scale 支持 FP32/UE8M0，必须是二维 `[tokens,num_scales]`，每 token 的 scale 字节数需 16-byte
对齐，当前预分配要求推导出的 scale block 至少为 128。普通输入在多节点时先 D2D 到预注册 staging；
外部 NCCL window tensor 可直接成为 LSA peer/Gin window 地址。非 CUDA Graph capture 路径当前会
在 dispatch 中同步 stream，读取实际接收数后再决定 staging copy 大小；capture 时无法读 host
计数，所以复制完整的最大预算。Local Permute 完全在 device 上读取计数，可跳过这个 host 同步。

### 9.5 Warp-specialized combine

expert MLP 完成后，HT combine 反向使用同一批映射：

1. 若输入不在外部 window，先复制到节点内可互访的 IPC staging。
2. 每张 GPU 读取本地 expert 槽，先做节点内部分归并。
3. rail GPU 把部分和通过 RDMA 发回 token 的 home node。
4. home node 再做跨节点和节点内归并，最终写 `[B,H]` 原始 token 顺序。

单节点和多节点会生成不同 kernel 变体；源码角色配置中单节点 combine 使用 6 个 warps，多节点
使用 11 个 warps。HT combine 不支持 FP8，但普通路径已支持 BF16、FP16 和 FP32。

### 9.6 公开 zero-copy 模式

`ncclEpGroupConfig_t::zero_copy` 有 AUTO/OFF/ON；当前 AUTO 解析为 OFF。HT 的 ON 会省略 group
内部 dispatch/combine token staging，因而 `dispatch_outputs.tokens` 与 `combine_inputs.tokens`
都必须绑定用户注册的 NCCL symmetric window，否则返回 `ncclInvalidArgument`。同节点 writer
通过 LSA peer pointer 直写用户窗口，跨节点使用同一 window/offset 做 GIN 地址；概率和部分
内部元数据仍可使用库自己的注册区，所以“zero-copy”特指主要 token staging，不代表零中间数据。

## 10. HT 训练前向与反向

前向过程是：

1. FWD dispatch 输入 token 和 router weight，输出送给本地 expert 的 token/权重/索引。
2. 框架运行 expert MLP。HT 要求框架在 combine 前应用 top-k 权重：Flat 中对一个接收 token
   的本地专家结果预归并；Expert Major 中每个槽只对应一个专家，逐槽乘其一维权重。
3. FWD combine 只接收处理后的 token，不接受 `inputs.topk_weights`，返回原顺序结果。

反向重用前向 handle 缓存的路由：BWD dispatch 将 token gradient 发向相同 expert 槽，不再输入
前向权重；BWD combine 同时接收 expert token gradient 和接收槽上的 weight gradient。实现先把
稀疏 weight gradient scatter 成 dense expert probability buffer，让 HT combine 一并归并，再按
缓存的 `topk_idx` gather 回 `[B,K]`。LL 当前拒绝 `NCCL_EP_BWD_PASS`。

## 11. LL/HT JIT 为什么存在

EP kernel 对节点数、LSA 大小、layout、数据类型、hidden、pipeline stage 数、是否 backward 等
参数高度特化。LL 还特化量化模式、纯 NVLink、top-k 索引宽度，并使用 cooperative/cluster
launch。全部预编译会产生巨大二进制和编译时间，因此 `device/jit/` 在首次遇到变体时：

1. 生成包含特定模板参数的 CUDA 源码。
2. 调用 `nvcc --cubin`；编译器优先取 `NCCL_EP_JIT_NVCC`，其次 `NVCC`、CUDA 路径和 PATH。
3. 按编译器、选项和 header tree 指纹缓存 cubin。默认目录为 `/tmp/nccl_ep/jit`，可用
   `NCCL_EP_JIT_CACHE_DIR` 修改。
4. 用磁盘锁避免多进程重复编译，并记录失败；进程内还有线程安全的内存 cache。
5. 通过 CUDA Driver API 加载 module、取得 function 并 launch。

因此 LL 和 HT 第一次运行某 shape 都可能有明显冷启动。运行环境不仅要有 `libnccl_ep.so`，还要
能找到构建时 staging 的 JIT headers 和可用 nvcc。路径可通过 `NCCL_EP_HOME`/`NCCL_HOME` 或
更细的 `NCCL_EP_JIT_SOURCE_DIR`、`NCCL_EP_JIT_BUILD_INCLUDE_DIR`、
`NCCL_EP_JIT_CUDA_INCLUDE_DIR` 覆盖；`NCCL_EP_JIT_LOG` 可诊断命中/编译情况。生产部署应把 cache
放在稳定、可写且节点本地的目录，并预热常用 shape。

## 12. 环境变量与调优边界

`nccl_ep_env.cc` 在每次 `ncclEpCreateGroup` 时读取一次环境，之后 handle 使用 group 内缓存，
运行中修改环境不会改变已有 group。布尔值只接受 `1/on/true` 或 `0/off/false`（不分大小写）。

| 环境变量 | 当前作用 |
|---|---|
| `NCCL_EP_ENV_VERBOSE` | rank 0 打印全部 EP 环境项及是否设置 |
| `NCCL_EP_DEBUG` | 输出 EP 版本 banner |
| `NCCL_EP_TIMEOUT_MS` | 覆盖 `config.timeout_ns`，用于 GPU 等待超时 |
| `NCCL_EP_COMM_SMS` | dispatch/combine 通信主 kernel SM 数 |
| `NCCL_EP_PROLOG_EPILOG_SMS` | HT permute/dup/reduce 辅助 kernel SM 数 |
| `NCCL_EP_PREPROCESS_NUM_SMS` | HT metadata scan SM 数 |
| `NCCL_EP_TOKENS_PER_CHUNK` | HT token chunk，向上对齐到 32 |
| `NCCL_EP_HT_EM_LOCAL_DUP` | 强制 HT EM 目标端本地 fan-out |
| `NCCL_EP_HT_EM_NVLINK_DUP` | 强制 HT EM 发送端 NVLink fan-out |
| `NCCL_EP_DISABLE_GUARD` | 关闭跨轮 LSA/RDMA staging 防覆盖保护，风险较高 |

JIT 相关的 `NCCL_EP_JIT_NVCC`、`NCCL_EP_JIT_CACHE_DIR`、`NCCL_EP_JIT_LOG` 和路径变量由
JIT runtime 直接解析，不在上述 per-group 配置表里。

## 13. 容量、内存与 CUDA Graph

### LL RDMA buffer

- `rdma_buffer_size=NCCL_EP_AUTO`：首个 handle 按实际 layout/top-k 集体分配；之后更大需求会
  barrier、注销 window、释放、重分配、清零和重新注册。
- 显式正值：group 创建时固定分配；handle 放不下返回 `ncclInvalidUsage`，不会扩容。
- AUTO 扩容会让已 capture graph 中烘焙的 base pointer 失效，也会丢失 staged operation 数据。
  扩容后必须重建 graph，扩容前必须完成所有 `send_only`。
- 所有 rank 必须以相同顺序、相同 layout/top-k 创建可能触发扩容的 handle。

### HT 静态预算

当前 v0.1 需要正的 `max_dispatch_tokens_per_rank` 和 `max_recv_tokens_per_rank`，后者至少不小于
前者。Expert Major 还要预算一个 token 同时命中多个本地专家以及 alignment padding 的最坏
情况。README 中“HT 可令 max dispatch 为 AUTO、先查询实际接收数再分配”的文字描述的是计划
能力；`ncclEpCreateGroup` 当前源码会直接拒绝 0，`ep_enums.h` 也已注明尚未支持。

HT 现在把 `max_dispatch_tokens_per_rank` 向上对齐到 chunk 作为 RDMA stride，不再在 group 创建时
检查旧的 8192 上限。默认多节点 chunk=64；单节点取 `4 * comm_sms` 后向上对齐到 32。可用
`NCCL_EP_TOKENS_PER_CHUNK` 覆盖，非 32 倍数会告警并向上取整。

## 14. 最小使用骨架

下面省略错误检查和具体 cudaMalloc，只展示对象关系：

```cpp
ncclEpGroupConfig_t gc = NCCL_EP_GROUP_CONFIG_INIT;
gc.algorithm = NCCL_EP_ALGO_LOW_LATENCY; // 或 HIGH_THROUGHPUT
gc.num_experts = 64;
gc.max_dispatch_tokens_per_rank = 128;
gc.max_recv_tokens_per_rank = 128 * nRanks; // HT 请按 layout 的最坏槽数预算
gc.max_token_bytes = hidden * sizeof(__nv_bfloat16);
gc.rdma_buffer_size = NCCL_EP_AUTO;
gc.num_qp_per_rank = NCCL_EP_AUTO;
gc.num_channels = NCCL_EP_AUTO;
gc.max_num_sms = NCCL_EP_AUTO;
gc.zero_copy = NCCL_EP_ZERO_COPY_OFF; // ON 时按算法要求提供 window-backed tensor

ncclEpGroup_t group;
ncclEpCreateGroup(&group, comm, &gc);

size_t route_shape[2] = {batch, topk};
ncclEpTensor_t route = NCCL_EP_TENSOR_INIT;
route.ndim = 2;
route.datatype = ncclInt64;
route.data = d_topk_idx;
route.sizes = route_shape;

ncclEpHandle_t handle;
ncclEpCreateHandle(&handle, group, layout, &route,
                   &layout_info, nullptr, stream);

ncclEpDispatch(handle, &dispatch_inputs, &dispatch_outputs,
               &layout_info, &dispatch_config, stream);
run_local_expert_mlp(...); // 上层负责，并遵守所选 layout 的加权规则
ncclEpCombine(handle, &combine_inputs, &combine_outputs,
              &combine_config, stream);

ncclEpHandleDestroy(handle);
ncclEpGroupDestroy(group);
```

所有 `dispatch_inputs`、`outputs`、config 和 layout info 同样要用对应 INIT 宏。真实、可运行的
shape 构造应参考 `contrib/nccl_ep/ep_test.cu`，不要仅凭上面骨架猜测可选 tensor。

## 15. 构建、运行与测试

先构建带 Device API 的 NCCL，再单独构建 EP：

```bash
make -j src.build BUILDDIR="$NCCL_HOME"
make -C contrib/nccl_ep MPI=1 BUILDDIR="$NCCL_HOME" \
  NVCC_GENCODE="-gencode=arch=compute_90,code=sm_90"

mpirun -np 8 "$NCCL_HOME/test/nccl_ep/ep_test" -a ll -t 50 -d 7168
mpirun -np 8 "$NCCL_HOME/test/nccl_ep/ep_test" -a ht -L fl -t 50 -d 7168
```

多节点通常还需正确配置 RDMA，并按 README 建议设置 `NCCL_GIN_TYPE=3`。当前 README 的正式
前提是 CUDA 13+、NCCL 2.29+、Hopper/Blackwell；Makefile 本身仍保留 CUDA 11.8+ 的 sm90
构建分支。仓内 `bindings/nccl4py/nccl/ep` 已删除，Python EP binding 改为独立发行，不能再按
旧文档使用 `nccl4py[cu13]` 假定它随 nccl4py 安装。

独立测试套件至少需要 4 张 GPU，不使用 MPI。脚本现在还会运行 zero-copy 测试，并把受影响的
HT Expert Major suite 分别在 Local Dup、NVLink Dup 模式下重跑：

```bash
make -C contrib/nccl_ep/tests \
  NCCL_HOME="$NCCL_HOME" NCCL_EP_BUILDDIR="$NCCL_HOME"
NCCL_HOME="$NCCL_HOME" NCCL_EP_BUILDDIR="$NCCL_HOME" \
  bash contrib/nccl_ep/tests/run_tests.sh 4
```

## 16. 如何选择 LL、HT 和 layout

- decode、小 batch、首 token/单步延迟最重要：优先 LL。
- prefill、训练、大量 token、需要充分利用跨节点带宽：优先 HT。
- 想让 expert kernel 直接消费连续专家分区：Expert Major；但为复制和 padding 预留空间。
- 想减少同 rank 多专家导致的 token 复制，且框架已有本地路由/归并能力：LL Rank Major 或
  HT Flat。
- 需要框架自动求导 router 权重：当前只有 HT backward 路径。
- 需要 `send_only` 分阶段重叠：当前只有 LL。

性能调优顺序建议是：先保证容量、layout 和 zero-copy window 正确，再固定常用 hidden/top-k
预热 JIT，然后用 `ep_bench` 比较算法、SM、chunk 和 QP。group config 的 `max_num_sms` 是三类
SM 的共同基线；AUTO 时 HT 主通信 kernel 默认 16 SM，LL 默认用全部 device SM，
prolog/epilog 和 preprocess 也默认用全部 SM。`NCCL_EP_COMM_SMS`、`NCCL_EP_PROLOG_EPILOG_SMS`、
`NCCL_EP_PREPROCESS_NUM_SMS` 可分别覆盖通信、Local Dup/Permute/Reduce、路由 scan，范围必须在
`[1,device_sm_count]`。`NCCL_EP_ENV_VERBOSE=1` 会在 group 创建时打印所有 EP 环境项。
`num_channels=AUTO` 仍解析为 10，但在当前 kernel 调用链中没有发现后续消费点，不应把它当成
有效调优旋钮。

## 17. 源码审阅发现的限制与易错点

1. **这是 v0.1 contrib API，且项目已迁移。** 它单独构建，公开 struct 做 size/magic 检查，
   许多参数错误使用 C/CUDA assert，可能直接终止进程；新开发发生在 `nccl-extensions`。
2. **专家必须平均切分。** 代码用整数除法求 `num_local_experts`，应显式保证
   `num_experts % nRanks == 0`。
3. **LL 实际 top-k 上限为 9。** `common.hpp` 的通用上限是 32，但
   `ll_ep_adapter.cu` 的 dispatch 上限和 `ll_combine_jit.cuh` 的 combine 上限都是 9。
4. **HT 动态容量尚未落地。** README 部分段落仍说支持 AUTO，当前 create 源码和 enum 注释
   明确相反，应以代码为准。`ep_test -m` 仍会把 max dispatch 设为 AUTO，因而也会撞上 create
   的 assert；不要把该选项当作可工作的动态容量示例。
5. **头文件有重复/过期声明注释。** `ncclEpHandleMemSize/InitHandle/UpdateHandle` 在当前头文件
   重复声明，且一处说 HT `num_topk` 可选/可传 `-1`，实现实际要求大于 0。
6. **数据类型不完全对称。** 普通 dispatch/combine 已支持 BF16/FP16/FP32；dispatch 还支持 FP8，
   combine 不支持。HT EM 的 Local Permute/Local Dup 路径也不能接 FP8。
7. **shape 具有硬约束。** hidden 需满足 16-byte/TMA 对齐，LL 还要求 128 的倍数；
   `(max_dispatch_tokens_per_rank * num_local_experts) % 4 == 0`。错误多由 assert 暴露。
8. **HT Forward combine 不替你乘权重。** 权重必须在 expert rank 上预先应用；LL Expert
   Major 才是在 combine kernel 中乘。
9. **`send_only` 不能无限排队。** 每 handle 只有一个 continuation；先 Complete 再复用。
10. **Graph 与扩容存在生命周期耦合。** LL buffer 变址后旧 graph 必须销毁重抓；HT capture
    会按最大容量复制而非实际计数。
11. **zero-copy 不是任意 tensor 免拷贝。** HT ON 要求 dispatch output/combine input 都绑定窗口；
    LL ON 目前只支持纯 NVLink Rank Major 非 FP8 dispatch。外部 window 在 kernel 完成前不能注销。
12. **默认 guard 有正确性意义。** 它防止相邻 round 的 staging WAR；除非框架已有等价同步，
    不要为省开销设置 `NCCL_EP_DISABLE_GUARD=1`。
13. **冷启动依赖 nvcc。** LL/HT 新变体都会运行时编译；只有 CUDA runtime、没有编译器或 JIT
    headers 时，首次运行会失败。

## 18. 建议的源码阅读顺序

1. `include/ep_enums.h`：先理解算法、layout 和权重责任边界。
2. `include/nccl_ep.h`：看 group/handle/tensor 和每个输入输出 shape。
3. `ep_test.cu`：把 API 串成一次完整执行。
4. `nccl_ep.cc` 的 `ncclEpCreateGroup`、`ncclEpUpdateHandle`、`ncclEpDispatch`、
   `ncclEpCombine`：理解 host 侧资源和分流。
5. `device/ll_ep_adapter.cu` → `device/jit/ll_*_jit.cuh` → `device/ll_ep.cuh`：沿参数特化、
   JIT entry、send/signal/recv/reverse combine 阅读 LL。
6. `device/scan_kernel.cuh`：理解 HT 如何把稀疏 top-k 编译成槽位映射和 EM fan-out 映射。
7. `device/hybridep_adapter.cu` → `device/jit/ht_*_jit.cuh` → `device/hybrid_ep.cuh`：看
   G2S/S2G/N2N warp、guard、Local Dup/Permute 和分层归并。
8. `device/jit/jit_runtime.cc`：理解两种算法共用的缓存、nvcc、module 和 launch 属性。

## 19. 总结

NCCL EP 的本质是一个“懂 MoE 路由的双向数据交换引擎”。LL 以更直接的 peer/RDMA 通路和
双缓冲换取小 batch 低延迟；HT 先把 top-k 编译成映射，再用节点内聚合、跨节点 GIN 和
warp-specialized TMA pipeline 换取大 batch 吞吐。dispatch 保存的不只是 token，还保存了
combine 所需的回程地址，这就是两者必须共用 handle 的根本原因。正确使用它的关键不只是调用
API，而是选对 layout、明确“谁乘 top-k 权重”、预留最坏容量，并遵守 collective、stream、
window、JIT 和 graph 的生命周期约束。
