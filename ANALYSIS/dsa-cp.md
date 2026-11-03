# DSA CP 代码理解与页面口径

分析基线：`xLLM-AI/xllm` main。核心文件 `xllm/python/model_executor/dsa_cp.py`（136 行，全部逻辑都在里面）。

## 它是什么

模块 docstring：`TP-token DSA CP with replicated KV, matching vLLM Ascend's legacy DSACP.`

即：对齐 vLLM-Ascend 旧版 DSACP 的语义，不是新设计。切的是 **query token 行**，不是 KV。

## 切片计划

`build_dsa_cp_context(num_tokens, q_lens, world_size, rank)`：

- `per_rank = ceil(num_tokens / world_size)`，rank r 取 `[r*per_rank, (r+1)*per_rank)` 与 `sum(q_lens)` 的交集。
- 对每个请求记录两个量：`local_q_lens[i]`（该请求落在本 rank 的行数）和 `query_suffix[i]`（该请求在本 rank 之后被切走的尾部行数）。
- 补齐只在出口 `restore_tp_heads` 之前加，回来后 `received[:num_tokens]` 截回。

## 因果长度

`local_context_lengths(global_lengths)` 有两条分支：

- **全部 q_len == 1（decode）**：用 `torch.arange` 在设备上构造 owner mask，`torch.where(active, global_lengths, 0)`。
  注释写明原因：decode 图的行是固定的，但 KV 长度每次回放都变，所以不能把 host 常量拷进图。
- **其他（prefill / chunked prefill）**：`lengths = (global_lengths - query_suffix).clamp_min(0)`，再把 `local_q_lens == 0` 的请求置 0。

## 换轴

`restore_tp_heads(local_output)`：`[local_tokens, all_heads, width]` → pad 到 `tokens_per_rank` →
`view(tokens_per_rank, world_size, local_heads, width).permute(1,0,2,3)` → `all_to_all(…, "tp")` →
`[global_tokens, local_heads, width]` → 截回 `num_tokens`。

换完之后 O 投影是原来的 row-parallel 路径，与关闭 CP 时一致。

## 生命周期

`get_dsa_cp_runtime` 把 `DsaCpRuntime` 缓存在 `get_forward_context().layer_shared_cache`，**不是**跨 context 复用。
代码注释给的原因：图入口会在 warmup 和 capture 之间复用 execution context；缓存在 ForwardContext 才能保证
长度算术被重新捕获，并在每次回放时读常驻的设备端长度。

ACL 图分支：`q_lens = [1] * dsa.seq_lens_q.numel() if dsa.is_acl_graph else dsa.seq_lens_q.tolist()`。
注释写明 `.tolist()` 会在 capture 期触发同步。

## 接入点（deepseek_v41.py）

```python
if cfg.enable_dsa_cp and cfg.cp_size > 1:
    raise ValueError("enable_dsa_cp cannot be combined with cp_size > 1")
self.enable_dsa_cp = cfg.enable_dsa_cp and tp > 1
if self.enable_dsa_cp and cfg.n_heads % tp:
    raise ValueError("enable_dsa_cp requires n_heads divisible by tp_size")
num_heads = cfg.n_heads if self.enable_dsa_cp else cfg.n_heads // tp
```

权重加载侧对应：`("attn.wq_b", "self_attn.q_b_proj", None if attn.enable_dsa_cp else 0)` —— 开 CP 时 `wq_b` 不切分。

forward 里只切 `positions` 和 `qr`；`q_a_proj` / `kv_proj` 仍吃完整 `hidden`。

## 页面上标为「待确认」的一处

gflag 帮助文本（`kernel_config.cpp`）：

> Enable TP-group DSA query parallelism for Python DeepSeek-V4.1 prefill and chunked prefill.
> Requires cp_size=1; **decode keeps the original TP path.**

但 `dsa_cp.py` 有专门的 decode ACL 图分支，attention forward 的启用判断也不区分 prefill / decode。
**代码为准：decode 同样走 CP。** 帮助文本疑似未随实现更新，页面上明确标注了这个不一致。

## 页面表达边界

- 不声称省显存：KV 整份复制，`wq_b` 每 rank 一整份，显存是**增加**的。
- 不声称降低注意力总 FLOPs：按头切和按行切的乘积相同。
- 只声称分摊「按 query 行计费」的那部分开销（逐行稀疏取数、metadata 组织、小算子下发），
  并明确说具体占比必须在目标配置上用 profiling 确认。
- **不给 before/after 数字**：手上的 `enable_dsa_cp` 实测绑定在另一代模型与平台组合上，口径不能平移到 Python V4.1。
- 提醒读者：开 CP 后每 rank 的 query 行很少，其他「按 token 行补齐」类优化会因此失去收益空间，两者不能相加估算。
