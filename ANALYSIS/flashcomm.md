# FlashComm1 页 口径（2026-11-06 重写）

分析基线：xLLM-AI/xllm main @ `543a1d2c8`。上一版页面写于 2026-10 下旬，结论已部分过期，这次从代码重写。

## 最重要的一条修正

上一版页面断言「Python 路径的 DeepSeek-V4.1 给 `--enable_flashcomm1` 不生效」，依据是当时建立
`FlashComm1ContextScope` 的只有 `models/llm/deepseek_v4.h` 和 `qwen3_next_hybrid_base.h` 两处。

**这条结论在 2026-11-05 之后不再成立。** `#686 feat: swa bounded replay & fc1 support.` 给 V4.1 加了
一条独立的 Python 实现。页面上保留了这条修正，并写明：写于 11-05 之前的 V4.1 profile，旧结论对那次测量仍然成立。

写页面时按 `git log --diff-filter=A` 核过首次引入时间，不是凭印象。

## 两条实现线是独立的

同一个 gflag 名，两套代码，门禁条件完全不同。页面开头就把这件事讲清楚，否则读者会把 C++ 线的
`−13.4%` 套到 V4.1 上。

| | C++ 线 | Python 线 |
|---|---|---|
| 载体 | `FlashComm1Context` + thread-local scope | `SpMetadata` + `EpSpTokenLayout`，显式传参 |
| 覆盖 | DeepSeek-V4、Qwen3-Next | DeepSeek-V4.1、Qwen3.5、Kimi-K3 |
| MoE 侧 | 同一个 tp_group 的 gather | 走 `moe_ep` 组的 gather / reduce-scatter |
| MMRS | 有，可选 | **没有**，`--enable_mmrs_fusion` 对它无效 |
| CP 组合 | **支持**（CP 外、FC1 内） | 不支持 |

## C++ 线新增的 CP 组合（页面第 5 节）

`FlashComm1TokenGeometry` 把原来混用的一个 token 数拆成三个。关键设计点，页面照写：

- 阈值判定用 `global_num_tokens`（CP 切分前），所以 CP 组内每个 rank 得到同一个开关决策；
- 几何用 `local_num_tokens`（CP 切分后），它成为 `ctx.original_num_tokens`；
- 组合门槛是 `min_local_num_tokens >= tp_world_size * 16`。原因：V4 的 CP 按序列连续切，段长不均，
  短 batch 会让高 cp_rank 只剩很少或 0 行。每个 rank 都能从全局 `q_seq_lens` 自己推出这个最小值，
  不需要通信，所以用它做门槛能保证整组要么全开要么全关。

CP 开着时 MMRS 被强制关掉（`enable_mmrs_fusion && !cp_active`），页面写了原因与「等测过再开」的建议。

## Python 线的三层门禁（页面第 5 节）

1. **启动期** `_validate_flashcomm1`：昇腾 W8A8、TP>1、CP=1、**DSA CP 关**、MoE TP=1、
   EP=DP×TP 且 DP-major rank 排布、全 MoE 层、阈值≥2。任一不满足就把开关置 False 并打 warning。
2. **拓扑与运行**：ACL 图 / `execution_state` / `is_spec_verify` 一律不走。
3. **每步 DP 投票**：活跃副本（`raw_counts > 0`）全是 prefill 才继续；一个真实 decode 副本否决整组；
   阈值取活跃副本的最大行数。空闲副本不投票但仍参与 EP 集合通信。

## 页面上特地点名的部署坑

**`enable_dsa_cp` 与 `enable_flashcomm1` 在 V4.1 上互斥。** 两个都传 true 时 FlashComm1 被静默关掉。
这条同时写进了 dsa-cp 页（新增一个红色 callout）和首页 dsa-cp 卡片的脚注，因为我们自己的对齐配置里
两个开关都在。启动日志判据：`V4.1 FlashComm1 disabled`。

## 页面表达边界

- C++ 线的 TTFT 数字（10211.2 → 8840.9 ms，−13.4%；其中序列分片 −11.8%、MMRS 再 −1.8%）照写，
  并把测试条件放在数字正上方的 callout 里。
- **Python V4.1 线不给任何数字**：#686 刚合入，我们没有绑定条件的 A/B。页面在「验证强度」表里把它
  明确列为「未验证」，并说明为什么不能平移 C++ 线的数字。
- 不声称 TPOT 收益。
- 加了「想自己测的话」一节，给可操作的对照方法和日志判据 —— 因为这个特性的失败模式是静默回退，
  不给判据读者会把「没触发」读成「没收益」。

## 写进页面的实现细节

- `kFc1LocalTokenAlignment = 16`，对齐到 `tp_world_size × 16`。
- `sp_row_parallel_projection` 的顺序是 matmul → reduce_scatter → **加 bias**。bias 在归并前加会变成 TP 倍。
- 量化 MMRS 的张量生命周期：`retain_quantized_mmrs_launch_tensors` 累计到 512 MB 才同步一次，
  最外层 scope 析构时也清一次。这是显存与同步次数的取舍。
- `FLASHCOMM1_ACTIVE` 日志**只打一次**，只能证明激活过，不能证明稳态每步都在走。

## 未写进页面

- `EpSpTokenLayout.from_token_counts` 的 `capacity_counts` / `valid_counts` 推导细节（对读者没有增量，
  页面只给结论：每个 (dp, tp) rank 的容量是 `ceil(count / tp_size)`）。
- Qwen3.5 / Kimi-K3 的接入差异（不在本站范围）。
