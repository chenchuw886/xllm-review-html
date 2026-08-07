const journey = document.querySelector("[data-journey]");
const tabs = document.querySelectorAll("[data-phase]");
const journeyNote = document.querySelector("[data-prefill-note] p");

const phaseCopy = {
  prefill:
    "Prefill 少等待 → Model Forward 更早结束 → LM Head 更早拿到 hidden states → Sampler 更早选出首 token → Incremental Decoder 更早输出文字 → <strong>TTFT 降低</strong>",
  decode:
    "后续 token 逐步进入 Decode 循环。FlashComm 的启用条件要求 Prefill，因此这里走原执行路径；<strong>TPOT 不是它的直接优化目标</strong>。",
};

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const phase = tab.dataset.phase;
    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    journey.classList.toggle("phase-decode", phase === "decode");
    journeyNote.innerHTML = phaseCopy[phase];
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
