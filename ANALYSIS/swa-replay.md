# SWA Bounded Replay 代码理解与页面口径

两个来源：

- xLLM 实现：`xllm/python/models/swa_bounded_replay.py`（190 行）
- 参考实现：DeepInfer（DeepSeek 自己的推理框架），`deepinfer/model_executor/input_metadata/yoco.py`、
  `deepinfer/worker/model_runner.py`、`deepinfer/model_executor/models/deepseek.py`、`examples/yoco/yoco_equiv.py`

## 架构前提

V4.1 的 CED：第 20 层是 decoder 唯一的 Full 层，第 21–39 层的 main KV 全部借自它，自己只算 Q 和滑窗 KV。
所以 prefill 时 consumer 层不需要逐行重算整个 prompt，只需要覆盖滑窗的一段后缀。

## xLLM 这版做了什么

常量：`SWA_BOUNDED_FIRST_CONSUMER = 21`、`SWA_BOUNDED_QUERY_ROWS = 140`。

`run_swa_bounded_consumers` 的步骤：

1. 按 `seq_lens_q` 取每个请求末尾 `min(n, 140)` 行，`index_select` 作用于 `hidden` / `pre_mix` / `positions` / `input_ids`。
   空 DP 副本保留 `indices = [0]` 的 dummy 行。
2. `replace(dsa, …)` 造 compact metadata：`seq_lens_q` clamp 到 140，`max_query_len` 取 min，
   `slot_mappings` 只对**行数等于原始行数**的平面做 index_select（压缩过的 source 平面行数更少，consumer 不改写它们）。
3. 稀疏索引重定基：`topk_idxs` 按 `drop` 整体减去 `drop`，负值保持；`candidate_blocks` index_select；`candidates` 切 `[drop:]`。
4. DP 对齐：`dp_all_gather` 交换各副本的**实际**行数（不能假设都是 140），再写回 `dp_execution_token_counts`。
5. 在 compact ForwardContext 里跑第 21–39 层，最后 `index_copy_` 散回原行号。

## 为什么是有损的

docstring 第一行：`Explicitly lossy V4.1 prefill consumer compaction.`
后面：`Missing consumer SWA dependencies make this approximate. Only suffix outputs are valid;
generation consumes the final row.`

即：xLLM 只做了「挑行 + 改长度」，**没有重算保留行的 SWA 可见域**。
被丢掉的行里有一部分本应落在保留行的滑窗可见域内，这部分依赖就真的丢了。

开启时会打 warning：
`Lossy SWA bounded replay enabled: retain 140 query rows in layers 21..39; prefix hidden states are approximate; decode uses full computation`

## DeepInfer 的无损做法（差别所在）

README：

> **Decoder SWA bounded replay:** reduce decoder prefill work by replaying only a bounded suffix.
> Set `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`
> for the default V4.1 settings; the retained suffix must cover the SWA window plus the maximum backtracking length.

`model_runner.py` 的断言给出 140 的来源：

```python
assert window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS <= DEEPINFER_YOCO_PREFILL_KEEP_TOKENS
```

两种模式，**可见域同式**（C++ 共用 `yoco_swa_visibility`），区别只在布局：

| 模式 | 布局 | 计算量 | 用途 |
|---|---|---|---|
| `mask` | full | 全量计算，矩形外可见域整行 -1 | 等价性基准 |
| `compact` | compact | 真截断，行数减少 | 生产路径 |

边界层 `cfg.n_prefill_layer` 上切 metadata 视图：`get_consumer_view()` / `get_mask_view()`，
把 `kv_swa_full_index_pivots` 换成对应布局的 pivots，consumer 层照常读。

**关键就是这个 pivots**：DeepInfer 重建了保留行的 SWA 可见域，xLLM 没有。

等价性验证：`examples/yoco/yoco_equiv.py`，L1 只比 metadata（不加载权重），L2 端到端比 logprob 与 top-K，
`--mtp` 额外比草稿头 main cache 写入等价性。

两边的结构约束一致：DeepInfer compact 要求草稿目标层全在边界同一侧，且 Engram 不在 consumer 层；
xLLM 的 `validate_swa_bounded_replay` 也要求 `engram_layer_ids` 不得 ≥ 21。

## 启用与门禁

gflag `enable_swa_bounded_replay`，`ModelConfig` 默认 `false`，注释：
`Lossy V4.1 prefill experiment; enabled only by explicit runtime opt-in.`

启动期 `validate_swa_bounded_replay` 在分配权重前校验，任一不满足抛 ValueError：
`n_layers == 40`、`window_size == 128`、`candidate_source_layer_id == 20` 且第 20 层 is_full、
第 21–39 层 `main_kv_source_layer` 全为 20、无 engram 层 ≥ 21。

运行期 `should_run_swa_bounded_replay`：
`acl_graph is not None` 或 `execution_state is not None` 时**不走**（decode 完全不受影响）；
`max(dp_execution_token_counts) <= 140` 时不走；其余情况**所有 DP 副本一起走**
（注释：只缩活跃 rank 会让 MoE gather 尺寸不一致）。

## 页面表达边界

- **首要信息是「有损」**，放在页面最顶部的红色 callout，不埋在后面。
- 明确写不支持 prefix logits。
- **不给收益数字**：我们没做过绑定条件的 A/B。只给省算上界的结构性描述（N ≫ 140 时，19 层的 query 行降到约 `140/N`）。
- **不给精度影响量化**：没跑过数据集，标为未验证。
- 把「补齐 pivots + 配等价性脚本」写成明确的下一步，而不是暗示它已经快好了。
- 上游状态如实写：encoder 侧已入 vLLM，decoder 侧未入；vllm-ascend 有实现但我们没测过收益。
