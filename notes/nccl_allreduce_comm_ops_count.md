# NCCL 一次 AllReduce 的最底层通信操作次数

> 结论来源:`src/enqueue.cc`(host 侧切片 `computeCollWork`)、`src/device/all_reduce.h`(设备 kernel
> `runRing`)、`src/include/collectives.h`(chunk/slice 步长常量)、`src/include/device.h`(`NCCL_STEPS`)。
> 默认讨论 **RING + SIMPLE** 算法(带清晰 channel/slice/step 结构);TREE / NVLS 的差异见末尾。

## 一句话

一次 AllReduce 被切成 `C=nChannels` 条并行通道,每条通道把 `S/C` 字节切成 chunk、用 `L` 个 loop 跑完,
每个 loop 做 `2*(nRanks-1)` 个 ring 步;每步本 rank 对相邻 rank 做一次 send + 一次 recv。因此最底层
"跨链路 chunk 搬运"次数为 **`C * L * 4*(nRanks-1)`**(send 与 recv 各 `C*L*2*(nRanks-1)`)。

---

## 什么是"最底层通信操作"

NCCL 最细的一次数据传输 = **一个 chunk 沿某条 channel 的 ring 链路走一步**(send 到 next / recv 从
prev)。在设备 kernel 里它是一次 `prims.directSend/directRecv/...` 调用;在网络层最终变成一次
`wrap_ibv_post_send`(一次 RDMA write / NVLink 拷贝)。这是计数基本单位。

- **chunk 级**:一次 `prims` 调用搬运一个 chunk(`chunkSize` 字节)← 本文主计数量。
- **slice 级(更细)**:每个 chunk 在网络层被切成 `chunkSteps=4` 个 slice,每个 slice 单独一次
  `wrap_ibv_post_send`(`enqueue.cc:2364` 的 `proxyOp->nsteps`)。这是绝对的"一次 RDMA write"。

---

## 三层分解

### ① Channel 维
AllReduce 被切成 `C = nChannels` 条并行通道,每条通道在自己那组 rank 上独立跑一遍 ring。

### ② Slice / Chunk 维
每条通道分到 `S/C` 字节,切成 chunk,每 chunk = `chunkSize` 字节:
- `stepSize = buffSizes[PROTO_SIMPLE] / NCCL_STEPS`(`NCCL_STEPS=8`,`enqueue.cc:2222`)
- RING+SIMPLE:`chunkSteps=4`、`sliceSteps=2`(`collectives.h:19-20`)→ `chunkSize = stepSize * 4`(`enqueue.cc:2225`)
- 每通道 loop 步长 `loopCount = nRanks * chunkCount`(`all_reduce.h:23`),即每个 loop 处理 `nRanks` 个 chunk
- 每通道 loop 数 `L = ceil( S / (C * nRanks * chunkSize) )`(`enqueue.cc:2361-2362`,
  `loopSize = nChannels * nRanks * chunkSize`)

### ③ Step 维
每个 chunk 在 ring 上走 `2*(nRanks-1)` 步(reduce-scatter `nRanks-1` + all-gather `nRanks-1`)。
`ncclPatternRingTwice` 的 `nstepsPerLoop = 2*(nRanks-1)`(`enqueue.cc:2347-2349`)。

设备 kernel `runRing`(`all_reduce.h:34-82`)每个 loop 对一条通道发起的 `prims` 调用数:

| 阶段 | 调用 | 次数 | 收发 |
|---|---|---|---|
| step 0 | `directSend` | 1 | send |
| 中间 `nRanks-2` 步 | `directRecvReduceDirectSend` | `nRanks-2` | 各 1 recv + 1 send |
| 第 `nRanks-1` 步 | `directRecvReduceCopyDirectSend` | 1 | 1 recv + 1 send |
| 再 `nRanks-2` 步 | `directRecvCopyDirectSend` | `nRanks-2` | 各 1 recv + 1 send |
| 收尾 | `directRecv` | 1 | recv |

→ **每 loop 每通道:send = 2·(nRanks−1),recv = 2·(nRanks−1)**。

---

## 计数公式(per rank)

```
L          = ceil( S / (C * nRanks * chunkSize) )
send次数    = C * L * 2*(nRanks-1)
recv次数    = C * L * 2*(nRanks-1)
链路搬运合计 = C * L * 4*(nRanks-1)      ← 最底层通信操作总数(send + recv)
ring步数    = C * L * 2*(nRanks-1)
```

- `S` = 总字节数
- `C = nChannels`(该 collective 使用的通道数)
- `chunkSize = stepSize * 4`(RING+SIMPLE)

网络 slice 级(更细,一次 RDMA write):
```
网络 send 次数 = C * L * 2*(nRanks-1) * chunkSteps   (= C*L*8*(nRanks-1))
```
`chunkSteps`/`sliceSteps` 只决定**流水线深度**(同时几个 chunk 在飞以隐藏延迟),**不改变数据总量与
chunk 级计数**。

---

## 数字验算(8 GPU / 1 node / 1 GB / RING+SIMPLE,C=16,chunkSize=512KB)

```
loopSize = 16 * 8 * 512KB = 64 MB
L        = ceil(1024MB / 64MB) = 16
send     = 16 * 16 * 2*7 = 3584
recv     = 3584
```

**校验数据量**:每 rank 实际发出 = 3584 × 512KB ≈ 1.75 GB。经典 ring all-reduce 每 rank 发送量 =
`2(nRanks−1)/nRanks · S = 2·7/8 · 1GB = 1.75 GB` ✓ 完全吻合,说明计数正确。

---

## 算法差异(TREE / NVLS)

分解框架(通道 × loop × 步 ×(收发))相同,但步数与 fanout 不同:
- **TREE**(`ncclPatternTreeUpDown`):每通道步数约 `2·log2(nRanks)`(fanout 树),每步多发多收
  (`NCCL_MAX_TREE_ARITY`)。
- **NVLS**:直接走 NVSwitch 多播,步数≈1,链路数取决于 `nvls.nHeads`。
- **LL / LL128**:`chunkSize` 按 `chunkSize/=2` 或 line-elem 折算(`enqueue.cc:2226-2227`),分解方法不变。

## 取真实数值
`nChannels`、`chunkSize` 运行时按拓扑/缓冲大小调出(`comm->nChannels`、`buffSizes[proto]`)。可开
`NCCL_DEBUG=INFO` 看 NCCL 打的通道数与 chunk 信息,或直接 dump `comm->nChannels` / `buffSizes`。
