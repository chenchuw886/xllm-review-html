const methodData = {
  draft: {
    icon: "AR",
    kicker: "经典方案",
    title: "小模型自回归起草",
    copy: "用同 tokenizer 的小模型逐个生成候选。通用、容易理解，但草稿长度 γ 越大，起草也要串行多走几步。",
    parallel: "低 · 随 γ 线性增长",
    dependency: "强 · 完整自回归",
    cost: "额外模型显存与串行 draft latency",
  },
  mtp: {
    icon: "N×",
    kicker: "多头 / 多模块预测",
    title: "MTP / Medusa",
    copy: "从 target 特征直接预测多个未来位置。额外模块通常较轻，但各位置若独立预测，块尾容易因缺少前序 token 条件而掉点。",
    parallel: "高 · 多位置同时预测",
    dependency: "弱到中 · 取决于具体结构",
    cost: "需要模型内置或额外训练多 token 模块",
  },
  eagle: {
    icon: "H→",
    kicker: "特征空间起草",
    title: "EAGLE 系列",
    copy: "在 hidden-state 空间逐步外推，再复用 target 的词表头；常与动态树候选结合。接受质量高，但自回归 draft 开销会随深度增长。",
    parallel: "中 · 候选树可并行验证",
    dependency: "强 · 特征级自回归",
    cost: "drafter 训练、串行起草与树验证开销",
  },
  dflash: {
    icon: "▦",
    kicker: "块扩散起草",
    title: "DFlash",
    copy: "用较深的并行 backbone 一次生成整个草稿块，起草延迟几乎不随 γ 增长；代价是位置间缺少已采样前缀依赖，尾部容易衰减。",
    parallel: "很高 · 整块一次 forward",
    dependency: "弱 · 块内双向但预测独立",
    cost: "需训练专用并行 drafter，低质量尾部会浪费验证",
  },
  dspark: {
    icon: "D✦",
    kicker: "半自回归起草",
    title: "DSpark",
    copy: "保留 DFlash 的并行主干，再用低秩 Markov/RNN head 轻量地串起块内 token。完整方案还用 confidence scheduler 按请求与负载裁剪验证长度。",
    parallel: "高 · 主干一次 forward",
    dependency: "中到强 · 轻量顺序 head",
    cost: "专用权重、draft KV/显存与调度工程",
  },
};

const methodTabs = [...document.querySelectorAll("[data-method]")];
const methodFields = {
  icon: document.querySelector("[data-method-icon]"),
  kicker: document.querySelector("[data-method-kicker]"),
  title: document.querySelector("[data-method-title]"),
  copy: document.querySelector("[data-method-copy]"),
  parallel: document.querySelector("[data-method-parallel]"),
  dependency: document.querySelector("[data-method-dependency]"),
  cost: document.querySelector("[data-method-cost]"),
};

methodTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const method = methodData[tab.dataset.method];
    if (!method) return;

    methodTabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });

    Object.entries(methodFields).forEach(([key, element]) => {
      if (element) element.textContent = method[key];
    });
  });
});

const acceptanceModes = {
  survival: {
    values: [83.9, 74.4, 65.7, 59.8, 54.7],
    note: "从整块起点观察，有 54.7% 的 proposal 能完整通过 5 个 draft token。",
  },
  conditional: {
    values: [83.9, 88.7, 88.3, 91.0, 91.4],
    note: "仅在前面位置均已通过的样本中计算；后四位约 89%–91%，说明 Markov head 没有出现明显的条件接受衰减。",
  },
};

const acceptButtons = [...document.querySelectorAll("[data-accept-mode]")];
const acceptBars = [...document.querySelectorAll("[data-accept-bar]")];
const acceptValues = [...document.querySelectorAll("[data-accept-value]")];
const acceptNote = document.querySelector("[data-accept-note]");

acceptButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const mode = acceptanceModes[button.dataset.acceptMode];
    if (!mode) return;

    acceptButtons.forEach((item) => item.classList.toggle("active", item === button));
    mode.values.forEach((value, index) => {
      acceptBars[index]?.style.setProperty("--value", `${value}%`);
      if (acceptValues[index]) acceptValues[index].textContent = `${value.toFixed(1)}%`;
    });
    if (acceptNote) acceptNote.textContent = mode.note;
  });
});

document.querySelectorAll("[data-copy]").forEach((button) => {
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(button.dataset.copy);
      const previous = button.textContent;
      button.textContent = "已复制";
      window.setTimeout(() => (button.textContent = previous), 1400);
    } catch {
      button.textContent = "请手动复制";
    }
  });
});

const navLinks = [...document.querySelectorAll(".topnav a")];
const observedSections = navLinks
  .map((link) => document.querySelector(link.getAttribute("href")))
  .filter(Boolean);

const observer = new IntersectionObserver(
  (entries) => {
    const active = entries
      .filter((entry) => entry.isIntersecting)
      .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!active) return;
    navLinks.forEach((link) => {
      link.classList.toggle("active", link.getAttribute("href") === `#${active.target.id}`);
    });
  },
  { rootMargin: "-20% 0px -68%", threshold: [0.08, 0.25, 0.5] },
);

observedSections.forEach((section) => observer.observe(section));
