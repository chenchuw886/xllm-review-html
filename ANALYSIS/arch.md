# DeepSeek-V4.1-Flash 架构页 口径

这一页不是特性页，是其余五页的术语基础。内容来自官方 reference 实现、tech report、vLLM / vllm-ascend 的模型实现，
以及 xLLM `xllm/python/models/deepseek_v41.py`。

## 页面上的硬事实

- 主干 552B + Engram 查表 196B；每 token 激活 prefill ≈ 8B、decode ≈ 16B；1M 上下文；原生图文。
- 主干 40 层，论文称 Causal Encoder-Decoder（CED）。**仍是 decoder-only，代码里 40 层顺序执行。**
- 四种层角色（按「全局 KV 怎么来」）：
  - SWA-only：第 0、1 层
  - Full：第 2、8、14、20 层（**只有这 4 层持有 main KV**）
  - Reindex：第 24、28、32、36 层（借第 20 层 KV，自己重跑 indexer）
  - Reuse：其余 28 层
- 实现口径：`kv_source_layer_ids=[2,8,14,20]`、`index_source_layer_ids=[2,8,14,20,24,28,32,36]`、
  `compress_ratios`（43 项，末 3 个 0 是 DSpark 三个草稿层占位）。规则是每层找层号不大于自己的最近 source 层。
- 压缩比：encoder 三个 Full 层 ratio 2；第 20 层 ratio 1。
- 稀疏注意力：窗口 128 + Top-512 压缩位，每 query 约看 640 个位置。indexer 32 头、head_dim 128。
- 分层稀疏索引：第 20 层建候选池（每 8 位置一块，取块内最高分，保留 Top-2048 块，≤16384 候选）；
  Reindex 层只在池内选 Top-512；Reuse 层不跑 indexer。
- mHC：残差流 4 条副本，`tokens × 4 × 5120`；4×4 混合矩阵 Sinkhorn 迭代 20 次。
  V4.1 的 Single-Pass 把折叠系数留给下一个子块用，论文说融成 Mega-mHC 后激活访存从 `(4n+4)d` 降到 `(2n+2)d`。
- Engram：2、3、4 阶 n-gram；压缩词表约 9.9 万；8 个哈希头；196B 参数分 2 个模块放在第 1 层和第 14 层；表项 FP8。
- 视觉：32 层 ViT，patch 14，2D-RoPE，3×3 pixel-unshuffle 降到 1/9，2 层 MLP 对齐主干维度。

## 为什么叫 CED（页面给的推导）

论文说 decoder 的全局 KV 来自 encoder 末层 hidden。看实现：第 20 层是 decoder 唯一 Full 层，
它的输入正好是第 19 层（encoder 末层）的输出；第 21–39 层全部复用第 20 层 KV。
**顺序执行就满足论文定义，没有额外的投影分支。**

推理价值在 prefill：prompt 的大部分 token 只要跑前 20 层，decoder 只需回放最近若干 token 补滑窗 ——
这正是 [swa-replay](./swa-replay.md) 那个特性的前提。

## KV 字节数：两个口径必须分开

**论文口径（890 B/token）**：
main KV 一条 = 512 维 FP4(0.5B) + 32 个 FP8 scale = 288 B；indexer K 一条 = 128 维 FP4 + 4 B scale = 68 B；
每条合计 356 B。encoder 3 个 Full 层 ratio 2：`3 × 356 ÷ 2 = 534`；第 20 层不压缩 356。合计 890 B。
**不含滑窗 KV。**

**昇腾口径（推算，未实测）**：按 vllm-ascend 快照 `cache_config.py` 的缓存格式推算。

| 缓存 | A2/A3 | A5 |
|---|---|---|
| main KV | BF16 512 维 = 1024 B/条 | FP4 + 每 16 维一个 BF16 scale = 320 B/条 |
| indexer K | INT8 + 每条一个 FP16 scale = 130 B/条 | FP4 + 每 32 维一个 UE8M0 scale = 68 B/条 |
| SWA KV | 由运行配置决定（未核） | MXFP8 + 每 32 维一个 BF16 scale = 544 B/条 |
| 全局 KV/token | ≈ 2.9 KB（未计 main KV 的 scale） | ≈ 970 B |

**页面明确标为推算，并写「不要把 890 B/token 用于 A3 容量规划」。** A3 上约为论文数字的 3 倍多，A5 才接近 890。

## 页面上写进去的三条「别说」

1. 别说「它是 encoder-decoder 架构，所以要两套 KV」—— 只有一条顺序执行的层栈。
2. 别说「Reuse 层是推理时的近似」—— 训练时就是这个结构。
3. 别说「所有 indexer 都与长度无关」—— 4 个 Full 层的打分仍与长度线性相关。

另有一条关于单层数据流：别说「先算滑窗注意力再算全局注意力再合并」——
两路是**同一个内核的入参**，不是两次注意力求和。

## 写进页面的实战教训

Compressor 的跨 chunk 状态与压缩边界对齐会影响 chunked prefill 和 prefix cache。
**图模式下每步的稀疏注意力 metadata 必须在回放前刷新**；xLLM 移植时漏了这一步，
图里一直用 warmup 的 dummy 值，表现为只在图模式下输出乱码。
这条同时出现在 dspark 页（它和接受率掉到个位数同源）。

## 未写进页面的

- IndexCache、YOIO、HySparse 这几个相关工作名词（对读者没有增量）。
- 量化章节只保留官方权重格式与「昇腾按上面那张表走」，不展开 NVIDIA 线的硬件门控。
