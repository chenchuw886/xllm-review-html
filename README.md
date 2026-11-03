# xLLM 特性解读

面向开发者的 xLLM 推理特性可视化解读，重点在昇腾 NPU 上的 DeepSeek-V4.1-Flash。

站点：<https://chenchuw886.github.io/xllm-review-html/>

## 页面

| 页面 | 内容 |
|---|---|
| [`arch/`](./arch/) | **DeepSeek-V4.1-Flash 模型结构**。40 层 CED、CSA2 跨层共享 KV、分层稀疏索引、Single-Pass mHC、Engram、KV 缓存族与昇腾上的字节推算。其余页面的术语基础。 |
| [`flashcomm/`](./flashcomm/) | **FlashComm**。长 prefill 把 row-parallel 层后的 `AllReduce` 换成 `ReduceScatter`，每个 TP rank 只处理约 `S/TP` 行 token。 |
| [`dspark/`](./dspark/) | **DSpark**。块并行草稿 + Markov 条件头 + 置信度头；一次 decode step 的时序、两套 KV 缓存、逐位置接受率。 |
| [`dsa-cp/`](./dsa-cp/) | **DSA CP**。V4.1 稀疏注意力在 TP 组内按 query token 行切分，KV 整份复制，出口一次 all-to-all 换轴。 |
| [`swa-replay/`](./swa-replay/) | **SWA Bounded Replay**。CED 下 decoder 层 prefill 只回放末尾 140 行；对照 DeepInfer 的无损实现说明当前这版为什么有损。 |
| [`engram-offload/`](./engram-offload/) | **Engram 主机侧卸载**。int8 n-gram 表搬到 host 内存，融合 CPU 算子查表去量化，换 HBM 预算。 |

## 写作口径

每个特性页固定五段：

1. 一句话结论与适用条件（含「什么时候别开」）
2. 机制：输入 → 处理 → 输出，配图
3. 代码证据：文件与符号
4. 启用方式与约束
5. 收益与验证强度，外加风险与待确认

状态标签的含义：

- **已实现** — 代码在仓里
- **已实测** — 我们跑过并留了数据，页面上会写明硬件、并行配置、输入输出长度与并发
- **未验证** — 没跑过，页面上不写数字
- **有损** — 会改变数值结果

性能数字一律绑定测试条件，换条件不成立。没在当前配置下实测过的收益不写数字。

## 技术实现

零依赖静态页面，GitHub Pages 直接发布。

- `assets/site.css` — 共享设计系统（浅色默认，深色跟随系统或手动切换）
- `assets/site.js` — 只做两件事：记住配色选择、高亮当前小节的目录项。禁用 JS 页面照常可读。
- 图表全部是内联 SVG，用 CSS 变量取色，自动适配深浅色。

本地预览：

```bash
python3 -m http.server 8731 --bind 127.0.0.1
```

## 分析口径记录

页面结论的推导过程和取舍记录在 [`ANALYSIS/`](./ANALYSIS/)，按特性分文件。

---

个人整理，非官方文档。结论以 [xLLM-AI/xllm](https://github.com/xLLM-AI/xllm) 仓库代码为准。
