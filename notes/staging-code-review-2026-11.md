# NCCL upstream/staging 重点提交代码 Review（2026-11）

> Review 基准：本地 `master`（4e8c939a，同步自 upstream/master 3e46c089，NCCL 2.32.3）
> Review 对象：`upstream/staging`（147a9511，NCCL 2.32.6-1）中挑选的 5 个主题、共 9 个提交
> 本文只做代码 review 与机制分析，未将 staging 合入 master。

| 主题 | 提交 | 作者 | 日期 |
|------|------|------|------|
| net_socket 竞态条件修复 | `3c74619c` | Madeleine Thompson | 2026-10-30 |
| cross-clique LSA 支持修复 | `60e0ada5` | Kamil Iskra | 2026-10-14 |
| 对称 AllReduce avg kernel 选择修复 | `abc82df9` | Stanislav Bardyuk | 2026-09-25 |
| Rubin 调优系列 | `113ead2c` `68dacb71` `f4cff12a` `54f53031` `d6553864` | Keith Caton 等 | 2026-09-28 ~ 11-02 |
| LL/LL128 性能提升 | `7b08ca7f` `22ee0247` `06d94f7b` | Kamil Iskra | 2026-10-27 |

---

## 1. net_socket 竞态条件修复（3c74619c）

### 1.1 功能介绍与目的

TCP Socket 网络传输（`src/transport/net_socket.cc`）采用**多线程生产者-消费者模型**：

- **主线程**（NCCL proxy 线程）：在 `ncclNetSocketTest()` 中把一个请求切分成最多 `nSocks` 个子任务（task），通过 `ncclNetSocketGetTask()` 投递到 per-helper-thread 的任务队列，并轮询子任务完成情况；
- **辅助线程**（`persistentSocketThread`，每 `nSocks/nThreads` 个 socket 一个）：循环扫描自己的任务队列，调用 `ncclSocketProgress()` 在 socket 上收发字节，推进每个任务的 `offset`。

两者通过共享结构 `ncclNetSocketTask`（含 `used/offset/result` 字段）和 `ncclNetSocketTaskQueue`（含 `next` 环形游标）无锁交互。修复前的代码对其中若干字段使用**普通读写**，存在三处数据竞争（C++ 标准下是 UB，实际可能造成丢失唤醒、读到陈旧数据甚至数据损坏）。

### 1.2 修复前的三个竞态

**竞态 A：`queue->next` 丢失唤醒（lost wakeup）**

辅助线程外层循环开头读取 `mark = myQueue->next`，作为"已见最新任务"的水位线；扫描结束后若空闲，则睡眠在条件变量上，等待谓词 `mark != next || stop` 为真。问题在于 `next` 是普通 load：

- 主线程投递任务：先 release-store `r->used = 1`，再在持锁下更新 `next`（普通 store）并 notify；
- 辅助线程的 `mark = next` 若被编译器/CPU 重排（如下沉到扫描之后），可能出现"扫描未看到新任务（`used` 不可见），但 `mark` 却读到了新值"的组合 —— 谓词判定"没有新任务"，线程带着队列里未处理的任务入睡，直到下一次投递才被唤醒。对延迟敏感的 LL 协议，这会直接造成微秒级甚至毫秒级的停顿。

**竞态 B：主线程无同步读取 `sub->offset` / `sub->result`**

`ncclNetSocketTest()` 用普通 load 检查 `sub->offset == sub->size` 判完成。缺少 acquire 语义意味着：主线程看到 `offset == size` 时，辅助线程刚写入用户缓冲区的**数据字节可能尚未对主线程可见**。随后 `*done = 1` 上报完成、NCCL 把缓冲区交给 GPU 拷贝 —— 收到的是陈旧数据。这是实打实的正确性风险（在 x86 强内存模型上很难复现，在 ARM 等弱内存模型上风险更高）。

**竞态 C：完成后的 use-after-recycle 写**

辅助线程旧代码直接经指针反复读写 `r->offset`/`r->result`。当主线程观察到 `offset == size` 后会把 slot 的 `used` 置 0 并允许 `GetTask` 复用该 slot；若辅助线程此后还写 `r->result = ncclSuccess`（旧代码每次 progress 都写，包括成功路径），会破坏新任务的字段。

### 1.3 修复机制

统一改用 `COMPILER_ATOMIC_LOAD/STORE`（C++ `std::atomic` 的编译器封装）+ 明确的 acquire/release 内存序，并在注释中写清每一对配对关系：

1. **`mark` 的 acquire load**（配对 `GetTask` 中 `next` 的 release store）：保证"从某个 `next` 值开始的扫描"能看到该值所计数的一切任务 —— `next` 的 release store 之前的 `used=1` 与任务字段写入，都对 acquire 之后的扫描可见。消除竞态 A。
2. **`GetTask` 中 `next` 的推进改为 release atomic store**：原因是辅助线程对 `next` 的一次 load（外层循环顶部）**不持有 threadMutex**，互斥锁无法提供同步，必须靠原子语义（注释明确说明了这一点）。
3. **辅助线程先把 `size/offset` 拷贝到局部变量**再调 `ncclSocketProgress()`，结果通过**一次 release store** 写回 `r->offset`；错误时才写 `r->result`（成功路径不再写，slot 初值即 `ncclSuccess`）。`offset` 到达 `size` 之后**绝不再触碰 `r`**，消除竞态 C。
4. **主线程 `Test()` 中对 `sub->result`、`sub->offset` 的 acquire load**（配对辅助线程的 release store）：看到 `offset == size` 即保证辅助线程的所有副作用（缓冲区数据）可见，消除竞态 B。

### 1.4 点评

修复思路教科书级：不引入新锁，只补齐 happens-before 边。注释把每一对 acquire/release 的配对点写死，可维护性好。风险点：`ncclNetSocketGetRequest`/`GetTask` 中主线程对 `r->used == 0` 的检查仍是普通读写，依赖"只有主线程分配 slot"这一不变量成立。属于正确性必合入的修复，无行为变化，回归风险低。

---

## 2. cross-clique LSA 支持修复（60e0ada5）

### 2.1 背景概念

- **LSA（Load/Store Architecture）**：NCCL 2.32 系列引入的对称内存机制 —— 同一"LSA 集合"内每个 rank 把自己的 big VA（虚拟地址空间）通过 `cuMemExport`/`cuMemImport` 互相映射，形成一个 `lsaFlatBase` 平铺地址空间，GPU kernel 可直接 load/store 对端缓冲区（P2P symm memory），无需走代理。
- **LSA team**：参与对称绑定的 rank 子集，要求**世界排名连续**且**通信器内所有 team 等大**（默认取各节点 rank 数的 gcd，参数 `NCCL_LSA_TEAM_SIZE` 可覆盖）—— 这是 NVLS 多播绑定所必需的均匀性。
- **cross-clique（MNNVL）**：Rubin 世代多节点 NVLink（Multi-Node NVLink）下，P2P 访问可跨越 MNNVL clique 扩展到整个 NVLink fabric domain，即 LSA 可达范围**大于单节点**。

### 2.2 缺陷成因

genkernels 合入时，为了让 LSA 摆脱 team 的"连续、等大"约束，把对称内存映射的 rank 集合从 `lsaTeam` 换成了 `localRankToRank`（本节点的 local ranks）。这在两种旧场景下等价（LSA = 本节点或 MNNVL clique），但 **cross-clique 场景下 LSA 可达域是整个 NVLink fabric domain，跨节点的远端映射因此丢失** —— cross-clique P2P 直接失效。

### 2.3 修复机制：引入 "LSA domain" 两层模型

新增 `struct ncclDevrState` 中的概念分层（`src/include/dev_runtime.h`）：

```
LSA domain（域）= 本 rank 通过 load/store 指令可寻址的全部 communicator rank
  ├─ 平常 = 本节点 localRanks
  ├─ MNNVL 激活时 = MNNVL clique.ranks
  └─ cross-clique 使能时 = 同一 NVLink fabric domain（按 peerInfo 的 fabric clusterUuid 过滤整个 nRanks）
LSA team（队）⊂ LSA domain：domain 内满足"连续 + 全通信器等大"约束的子集，供 NVLS/MC 对称绑定使用
```

具体实现：

1. **`initLsaDomainData()`（新函数）**：一次性算出 `lsaDomainRankList/lsaDomainSelf/lsaDomainSize`。cross-clique 分支遍历所有 `peer`，比对 `peerInfo[].fabricInfo.clusterUuid`，同 UUID 的 rank 依升序进 domain 名单。诊断代码 `ncclDiagP2pBuildRankSet()` 原有的同款逻辑删除，改为直接复用 domain 数据（单点维护）。
2. **字段更名消歧**：`lsaSelf/lsaSize/lsaRankList` → `lsaTeamSelf/lsaTeamSize/lsaTeamRankList`。`lsaTeamRankList` 不再独立 malloc，而是**指向 `lsaDomainRankList` 内部的偏移**（`lsaDomainSelf - lsaTeamSelf` 处），析构时只 free domain 名单（注释特别提醒不要 double-free）。
3. **对称映射全部改按 domain 进行**（`symMemoryMapLsa()` 等）：`bootstrapIntraNodeAllGather/Barrier` 的 rank 集合、`lsaFlatBase` 平铺空间的槽数（`lsaDomainSize * bigSize`）、远端 handle 的 import/map 循环，一律从 `localRankToRank` 换成 `lsaDomainRankList`。cross-clique 的跨节点远端映射由此恢复。
4. **`ncclGetPeerDevicePointer()` 修复**：旧代码用 `rankToLocalRank[peer]` 换算平铺空间槽位 —— cross-clique 的 peer 根本不在 localRanks 表里。新代码复用 genkernels 基础设施 `lsaPeerDelta4GDev`（per-world-rank 的 4GB 粒度槽位差表，device 侧/host 侧各一份）：`peerDelta4GHost[peer] == INVALID` 则不可达返回 nullptr，否则 `add4G(lsaFlatBase + winOffset, delta) + offset` 直接算出对端指针。
5. **`computeLsaTeamSize()`**（原 `computeLsaSize`）逻辑不变：cross-clique 且单一 NVLD 时 team 扩展到全通信器，否则走节点 rank 数的 gcd。`ce_coll.cc` 等所有消费者改用 `lsaTeamSelf/lsaTeamSize`（CE 侧同步窗口、AllGather/AlltoAll/Scatter/Gather 语义均基于 team，不受影响）。

### 2.4 点评

这是典型的"概念模型跟不上硬件演进"的修复：genkernels 用 localRanks 代替 lsaTeam 解决了均匀性约束，却隐含假设"LSA 可达域 ≤ 本节点"。两层 domain/team 模型把"**地址可达性**"（domain，映射用）与"**绑定均匀性**"（team，NVLS/MC 用）解耦，是正确的长期抽象。改动面大（11 文件）但机械替换居多；需要重点回归 MNNVL 与 cross-clique P2P 两组拓扑。

---

## 3. 对称 AllReduce avg kernel 选择修复（abc82df9）

### 3.1 功能介绍与目的

对称内存 kernel（symmetric kernels，`src/sym_kernels.cc` + `src/device/symmetric/generate.py` 生成）为对称窗口上的集合通信提供免拷贝 fast path。调度入口 `ncclSymkImplemented()` 判定某 (collective, redOp, datatype) 组合是否有对称 kernel，有则被 symmetric scheduler 接管。

### 3.2 缺陷机制

自 2.30（提交 8df9d244）起，`ncclSymkImplemented()` 中 AllReduce 与 ReduceScatter 共享同一个 case 块，两者都接受 `ncclDevSumPostDiv`（即 `ncclAvg` 的设备表示）。但代码生成器只**为 ReduceScatter 生成 avg kernel**：

```python
# generate.py::enumerate_kernels()
for red in ["sum", "avg"]:
  for algo in [...AllReduce 算法...]:
    if red == "avg": continue        # AllReduce 不生成 avg kernel
    yield Rec(coll="AllReduce", ...)
  for algo in [...ReduceScatter 算法...]:
    yield Rec(coll="ReduceScatter", red=red, ...)   # 两种 redOp 都生成
```

于是 `ncclAvg` 的对称窗口 AllReduce 走上这条失败链：

```
ncclSymkImplemented(AllReduce, ncclDevSumPostDiv, float) → true（错误放行）
  → symmetric scheduler 接管
    → ncclSymkGetKernelIndex(AllReduce id, red=SumPostDiv, ty) → -1（表里没有）
      → 调度器用 -1 索引 kernel 表 → 越界
        → cudaErrorInvalidDeviceFunction / ncclUnhandledCudaError，所有 float 类型必现
```

对应 GitHub issue #2447。

### 3.3 修复机制（两处，双保险）

1. **根源**：`ncclSymkImplemented()` 的 AllReduce 分支改为仅接受 `ncclDevSum`（`return red == ncclDevSum && isFloat && ty != ncclFloat64;`），avg 退回常规 kernel —— 恢复 2.30 之前的行为；ReduceScatter 分支不再 fall-through，显式 `return false`。
2. **防御**：`ncclSymmetricTaskScheduler()` 对 `kernelIndex < 0` 增加 WARN + `ncclInternalError` 早退，杜绝今后任何"implemented 与生成表不一致"再演变成越界索引。

### 3.4 点评

小而漂亮的回归修复：根因（声明与生成表不一致）+ 症状（负索引）双管齐下。局限：没有补齐 avg AllReduce 对称 kernel（明确写在提交信息里），`ncclAvg` AllReduce 暂时退回常规路径，性能与 2.29 持平。验证方式：`all_reduce_perf -R 2 -c 1` 全 float 类型（修复前必挂、修复后通过），sum 与 ReduceScatter avg 仍走对称 kernel。

---

## 4. Rubin 调优系列（5 个提交）

### 4.1 背景

Rubin（`RUBIN_AND_LATER` 宏：`sm >= 107 && sm != 110 && sm != 120 && sm != 121`，Vera Rubin 平台）上 NVLink 带宽/拓扑（GBx: 4 GPU/主机 + 双 NIC）与 Hopper/Blackwell 差异显著，NCCL 新一代 pipeline cost model（`src/tuning/`）此前对 Rubin 的常数多为 `-1`（未标定，模型自动推算）。这一系列提交用实测数据标定 Rubin 常数并修正模型结构性偏差。5 个提交按逻辑顺序：

### 4.2 逐提交机制

**（1）`113ead2c` — LL/LL128/Tree 带宽上限常量化（模型结构修正）**

- 新增 `ncclTuningProtoBW()`（`src/tuning/model.h`）：把 pipeline model 估算的 per-CTA 带宽用调优常数表 clamp（`llMaxBws` / `perChMaxRingLL128Bws` / `perChMaxTreeLL128Bws` / `perChMaxTreeBws` / `perChMaxNVLSTreeBws`，按 compCap 索引）。常数有效（>0）才 clamp，保持向后兼容。
- **Rubin 语义变化**：LL/LL128 在新架构上是 **per-CTA 带宽**而非 per-channel 总上限，且 channel 内可多 CTA 叠加。为此 NVLS pipeline model（`nvls.cc`）把带宽除以 `ctasPerChannel` 再 clamp，保证多 CTA 时不重复计数。
- 填入 Rubin 实测值：LL `{2.0, 0.6, 0.46}` GB/s、Ring LL128 `{21.6, 17.0, 14.0}`、Tree LL128 `{11.2, 11.91, 11.91}`（N1/N2/N4）。

**（2）`68dacb71` — Rubin 上 P2P 网络 chunk 提到 512K**

- 三个 chunk 参数默认值从编译期常量改为 `-2`（自动），仅当用户显式设置（>0）才生效 —— 为平台差异化默认值铺路。
- `ncclComputeBuffSizes()`：GBx（ARM + compCap ≥ 100）每主机仅 4 GPU，聚合因子小，ratio 先 ×2；Rubin（`RUBIN_AND_LATER`）流量拆到两个 NIC，再 ×2 —— 多机默认网络 chunk 从 128KB → **512KB**，摊薄 per-chunk 同步开销。
- 附带把"P2P chunk 被 Simple buffer 上限截断"的情况从静默改为 `INFO` 日志。

**（3）`f4cff12a` — Vera Rubin NVLSTree chunk 尺寸策略**

- 旧代码 `chunkSize = treeMaxChunkSize = 512KB` 一刀切。新策略按拓扑缩放：
  - `ranksPerDomainFactor = pow2Up(DIVUP(32, minLocalRanks))` —— 每 domain rank 越少，单 chunk 越大（把每步 32 个"slot"填满）；
  - 多 domain（nNodes > 2）时 ×2，进一步放大管线深度；
  - 上限 512KB：`chunkSize = min(512KB, 64KB × ranksPerDomainFactor × multiDomainChunkFactor)`。
- 加 `INFO` 日志输出决策参数。TODO 注释言明：待 pipeline cost model 支持 NVLSTree 管线建模后迁回模型。

**（4）`54f53031` — 小消息 Ring CTA 数限制（平台回调钩子）**

- 新增平台扩展点 `ncclTuningPlatformChannelTuning()`（`src/tuning/platforms/platforms.h`），在 `ncclTuningGetChannels()` 主流程中调用 —— 平台可以最后裁决 CTA/通道数。
- Rubin 实现（`rubin.h`）：多机、消息 ≤ 2GB、选了 Ring 时，若 `minLocalRanks >= 56` 则 `maxChannels = min(maxChannels, 24)`，且不低于 `minCTAs` 下限。动机：大节点上小消息开满 CTA 只增加调度/同步开销，收益为负。

**（5）`d6553864` — Ring Simple 与 LL128 切换点标定**

- `hwLatencies[NVLINK][Ring][Simple]`：多机时直接设 17.3μs（覆盖单机 ≥64 卡时 +13μs 的旧补丁逻辑）；`hwLatencies[NET][Ring][Simple]` 从 150 → 219.5μs（实测网络启动延迟更高）。
- `perChMaxTreeLL128Bws` Rubin N2/N4：17.0/14.0 → 16.3/11.4（下调 Tree LL128 带宽上限）。
- 效果：cost model 在 Ring Simple 与 LL128 之间的切换点（crossover）向实测靠拢，中消息段不再错选协议。

### 4.3 点评

系列呈现出清晰的调优方法论：先给模型装"上限钳制"结构（1），再按硬件事实修正 chunk/CTA 等离散决策（2、3、4），最后用实测微调连续常数把切换点对齐（5）。风险集中在（4）的 `minLocalRanks >= 56` 与 2GB cutoff 这类魔法数——硬编码拓扑启发式，GBx 后续换代需要跟进。（2）把参数默认值改为 `-2` 也意味着用户显式设置 `NCCL_P2P_NET_CHUNKSIZE` 的语义不变，但 unset 时的值随平台变化，升级说明里值得提示。

---

## 5. LL / LL128 性能提升（7b08ca7f + 22ee0247 + 06d94f7b）

### 5.1 功能介绍与目的

LL（低延迟协议，`ncclLLFifoLine` 每行 16B：data/flag 交错，含 8B 有效数据）与 LL128（128B/线、16 个 64 位字中 15 个数据 + 1 个 flag，~94% 带宽效率）是 NCCL 的两个低延迟协议，热路径在 device 侧 `src/include/nccl_device/impl/flow__ll_funcs.h` / `flow__ll128_funcs.h` 的 `ncclFlowBase::process()`：每线程循环"load 用户数据 → reduce → 写共享 FIFO → 存用户数据"。三连击分别优化**数据搬运的对齐分支**（LL）、**LL128 的 16B 向量化分支**、以及**内层循环的代码生成质量**。

### 5.2 提交一 `7b08ca7f`：LL 的 4 字节对齐 fast path

**问题**：LL 数据装载器 `ncclFlowLLDataLoader` 对 `sizeof(T) < 4`（fp16/bf16/fp8 等）一律走"未对齐"路径：3 次 32 位读 + 两次 `__funnelshift_r` 字节桶形移位拼出 8B 数据。但很多场景源缓冲实际是 4 字节对齐的 —— 最典型的是 AllGather，早期就被改写为 uint8_t 单 kernel，此时逐字节移位纯属浪费。

**机制**：
- `process()` 外层**只算一次** `srcMisalign4 = src % 4`、`dstMisalign4 = dst % 4`，作为参数传入，替代热循环内逐次取地址算模。
- `begin(src, nElts, misalign4)`：`misalign4 == 0` 时直接两次 32 位 load（`u4[0]/u4[1]`）填满 8B，省掉 funnel shift；尾部允许 over-read —— 不跨对齐边界即不跨页，不会缺页（注释说明了安全性论证）。
- `ncclFlowLLStoreData(dst, data, nElts, misalign4)` 对称优化：整行（`nElts == EltPerLine`）且 4B 对齐时直接两个 32 位 store，不再逐元素。
- 旧代码 union 里 `u4[3]`（16 位类型跨 12B 窗口）与按 `sizeof(T) <= 2` 的分支统统简化为 `u4[2]` 双字窗口。

**效果**：对齐场景每 8B 数据省 2 次 funnel shift 与最多 1 次冗余 load；AllGather（uint8 化）受益最直接。

### 5.3 提交二 `22ee0247`：LL128 的 16 字节向量化分离

**问题**：LL128 装载/存储循环里对每一段数据做运行时判断 `addr % 16 == 0` 走 `ncclFlowLL128Load128/Store128`（128 位向量访问），否则退回 `LoadDataWord` 逐字路径。分支在循环体内反复求值，且两条路径的代码交织影响编译器生成质量。

**机制**：
- 同样把 `srcMisalign16`、`dstMisalign16`（及 `dstMisalign4`）提到 `process()` 外层算一次。
- `ncclFlowLL128LoadDataBegin(..., misalign16)` 重构为**两个独立循环**：`misalign16 == 0` 的循环体只含 128 位 load（尾部 over-read 同样有"不跨对齐边界不跨页"的安全性注释）；否则的循环体只含逐字路径。分支从"每段每线程"降为"每 chunk 一次"。
- `ncclFlowLL128StoreData(..., misalign16, misalign4)` 同理：`misalign16 == 0 && nElts % EltPer128 == 0` 走纯向量循环；混合情形才进入带内部分支的通用循环（其内部 16B 判断复用传入的 `misalign16`，4B 判断复用 `misalign4`）。
- 两处 `else` 分支补 `data[2g]=data[2g+1]=0`：非正确性必需，但帮助 warp 保持收敛（避免 divergent 分支拖慢同 warp 其他 lane）。
- 一个反直觉的细节被注释记录：`ncclFlowLL128LoadDataWord` 里**故意传 misalign4=1**（假设未对齐）而不是动态值 —— 实测在 AllGather_Ring_LL128 上，让代码生成器只生成单分支，比"0/1 双分支 + 动态选择"更快。性能工作中"实测优先于直觉"的典型案例。

### 5.4 提交三 `06d94f7b`：`process()` 内层循环代码生成优化

不改变算法，纯粹帮编译器生成更优 PTX，五项手段（LL/LL128 同步实施）：

1. **共享内存快照到局部数组**：`shmem.sendBuf/recvBuf/sendFlag/recvFlag` 在进入内层循环前拷贝到局部数组 `sendBuf[]/recvBuf[]/...`。共享内存访问无法被寄存器化，逐次 `shmem.sendBuf[peer]` 的 load 挡住了循环不变量外提；局部数组允许编译器放入寄存器（peer 数量是小编译期常量，`NCCL_FLOW_MAX_PEERS` 上限内可完全展开）。
2. **`isPost` 拆分为 `isPostSend`/`isPostRecv`**：旧的单布尔在收/发后处理各处重复混合判断；拆开后 `if (isPostSend) backend.postSend(...)` / `if (isPostRecv) backend.postRecv()` 各自独立，削减冗余判断。
3. **`if constexpr` 标注收/发独占代码**：`Send`/`Recv` 是模板编译期常量，内层"读对端 FIFO 归约"（仅 Recv）与"写发送 FIFO"（仅 Send）的循环用 `if constexpr` 包裹后，纯发送 kernel 不再生成（也不再求值）接收循环的任何指令。原来 `chunkRecvPeers == 0` 时还要靠运行时跳过的 `LoadDataFinish` 调用，现在编译期排布为 `else if (dataValid)` 直调。
4. **源/目的指针缓存与指针递增**：`src`/`dst` 在循环外取一次基址（`srcLambda(0) + eltOffset`），循环内 `src += eltIncrement` 递增，替代每轮 `srcLambda(0) + eltOffset` 重算（lambda 可能含多层间接）。
5. **循环控制简化**：以 `remaining > 0` 作循环条件递减计数，`sliceSteps` 等仅断言使用的量移入 `#ifndef NDEBUG`。

**效果**：内层循环访存指令占比提高、分支与共享内存 traffic 减少；对 LL/LL128 这类微秒级延迟协议，指令数的削减直接换算为时延。三提交均不改变协议语义与内存布局，风险集中在 over-read 安全性论证（不跨对齐边界 ⇒ 不跨页 ⇒ 无缺页）与编译器行为差异，需要全 dtype × 全 collective 的矩阵回归。

---

## 6. 总结与合入建议

| 主题 | 性质 | 影响面 | 建议 |
|------|------|--------|------|
| net_socket 竞态修复 | 正确性（UB） | 仅 TCP transport，多线程 socket 部署受益 | 建议随 2.32.6 一起跟进，低风险 |
| cross-clique LSA 修复 | 功能恢复 | MNNVL/cross-clique P2P；11 文件大改 | 用到 MNNVL 再跟进；需拓扑回归 |
| avg AllReduce 修复 | 回归修复（#2447） | 对称窗口 + ncclAvg 用户 | 低风险，建议跟进 |
| Rubin 调优系列 | 性能/调参 | 仅 Rubin（`RUBIN_AND_LATER`）路径 | 非 Rubin 硬件零影响，无紧迫性 |
| LL/LL128 性能三连击 | 性能 | 所有平台的 LL/LL128 device kernel | 通用收益，但需较大回归矩阵 |

总体判断：staging 上这批改动质量高、注释完善（尤其内存序配对与 over-read 安全性论证都写进了注释）。若近期跟进，建议等 NVIDIA 把 staging 合入 upstream/master（2.32.6 发布后惯例在数周内），随 master 常规同步拿到全部 298 个提交，避免手工 cherry-pick 造成的依赖断裂（LL 三连击之间存在明确的增量关系，Rubin 调优系列亦然）。
