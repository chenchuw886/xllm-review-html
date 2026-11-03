/* xLLM Feature Review — shared behaviour.
   Two jobs only: remember the light/dark choice, and highlight the TOC entry
   for the section currently on screen. Every page works without this file. */

(function () {
  "use strict";

  var root = document.documentElement;

  /* ---------- theme ---------- */

  function readStored() {
    try {
      return localStorage.getItem("xllm-review-theme");
    } catch (err) {
      return null;
    }
  }

  function store(value) {
    try {
      localStorage.setItem("xllm-review-theme", value);
    } catch (err) {
      /* private window or blocked storage: the page still renders correctly */
    }
  }

  function systemPrefersDark() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function currentTheme() {
    return root.getAttribute("data-theme") || (systemPrefersDark() ? "dark" : "light");
  }

  var stored = readStored();
  if (stored === "dark" || stored === "light") {
    root.setAttribute("data-theme", stored);
  }

  function label(button, theme) {
    var dark = theme === "dark";
    button.textContent = dark ? "☀" : "☾";
    button.setAttribute("aria-label", dark ? "切换到浅色模式" : "切换到深色模式");
    button.setAttribute("title", dark ? "切换到浅色模式" : "切换到深色模式");
  }

  var toggles = document.querySelectorAll("[data-theme-toggle]");
  Array.prototype.forEach.call(toggles, function (button) {
    label(button, currentTheme());
    button.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      store(next);
      Array.prototype.forEach.call(toggles, function (other) {
        label(other, next);
      });
    });
  });

  /* ---------- TOC highlight ---------- */

  var links = document.querySelectorAll(".toc a[href^='#']");
  if (!links.length || !("IntersectionObserver" in window)) {
    return;
  }

  var byId = {};
  var targets = [];
  Array.prototype.forEach.call(links, function (link) {
    var id = link.getAttribute("href").slice(1);
    var heading = document.getElementById(id);
    if (!heading) {
      return;
    }
    byId[id] = link;
    targets.push(heading);
  });

  var visible = [];

  function paint() {
    Array.prototype.forEach.call(links, function (link) {
      link.classList.remove("is-active");
    });
    if (visible.length && byId[visible[0]]) {
      byId[visible[0]].classList.add("is-active");
    }
  }

  var observer = new IntersectionObserver(
    function (entries) {
      entries.forEach(function (entry) {
        var id = entry.target.id;
        var at = visible.indexOf(id);
        if (entry.isIntersecting && at === -1) {
          visible.push(id);
        } else if (!entry.isIntersecting && at !== -1) {
          visible.splice(at, 1);
        }
      });
      // Keep document order so the topmost visible section wins.
      visible.sort(function (a, b) {
        return targets.findIndex(function (n) { return n.id === a; }) -
          targets.findIndex(function (n) { return n.id === b; });
      });
      paint();
    },
    { rootMargin: "-80px 0px -65% 0px", threshold: 0 }
  );

  targets.forEach(function (heading) {
    observer.observe(heading);
  });
})();
