if (typeof SphinxHighlight !== "undefined") {
  SphinxHighlight.highlightSearchWords = () => {};
}

document.addEventListener("DOMContentLoaded", () => {
  const query = new URLSearchParams(window.location.search).get("q")?.trim();
  const heading = document.getElementById("search-heading");
  if (heading && query) {
    heading.textContent = `Results for “${query}”`;
    document.querySelectorAll(".search-button__default-text").forEach((label) => {
      label.textContent = query;
    });
  }

  const scrollToResult = (url) => {
    const article = document.querySelector(".bd-article");
    if (!article) return;
    article.querySelectorAll("mark.search-match").forEach((mark) => mark.replaceWith(...mark.childNodes));
    article.normalize();
    article.querySelectorAll(".search-match-block").forEach((node) => node.classList.remove("search-match-block"));
    const anchor = url.hash ? document.getElementById(decodeURIComponent(url.hash.slice(1))) : null;
    const scope = anchor || article;
    const terms = (url.searchParams.get("highlight") || "").toLowerCase().split(/\s+/).filter(Boolean);
    const candidates = [...scope.querySelectorAll("p, pre, li, td, dt")];
    if (scope.matches("p, pre, li, td, dt")) candidates.unshift(scope);
    const score = (node) => terms.filter((term) => node.textContent.toLowerCase().includes(term)).length;
    const match = candidates.filter((node) => score(node) > 0).sort((a, b) => score(b) - score(a))[0];
    if (match) {
      const pattern = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi");
      const walker = document.createTreeWalker(match, NodeFilter.SHOW_TEXT);
      const texts = [];
      while (walker.nextNode()) texts.push(walker.currentNode);
      for (const text of texts) {
        const parts = text.textContent.split(pattern);
        if (parts.length === 1) continue;
        const fragment = document.createDocumentFragment();
        parts.forEach((part, index) => {
          if (index % 2) {
            const mark = document.createElement("mark");
            mark.className = "search-match";
            mark.textContent = part;
            fragment.append(mark);
          } else fragment.append(document.createTextNode(part));
        });
        text.replaceWith(fragment);
      }
    }
    const destination = match || anchor;
    const firstContent = article.querySelector("p, pre, ul, ol, table");
    const atTop = !destination || destination === firstContent
      || destination.matches("h1") || destination.querySelector(":scope > h1");
    requestAnimationFrame(() => {
      if (atTop) window.scrollTo({ top: 0, behavior: "instant" });
      else {
        destination.classList.add("search-match-block");
        destination.scrollIntoView({ behavior: "instant", block: "start" });
      }
    });
  };

  document.addEventListener("click", (event) => {
    if (event.defaultPrevented || event.button !== 0
      || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest("#search-results a");
    if (!link || link.target === "_blank" || link.hasAttribute("download")) return;
    const url = new URL(link.href);
    if (url.origin !== location.origin || url.pathname !== location.pathname) return;
    event.preventDefault();
    const dialog = document.getElementById("pst-search-dialog");
    if (dialog?.open) dialog.close();
    if (url.href !== location.href) history.pushState(null, "", url.href);
    scrollToResult(url);
  });

  const restoreResult = () => {
    const url = new URL(location.href);
    if (url.searchParams.has("highlight")) scrollToResult(url);
  };
  window.addEventListener("load", restoreResult);
  window.addEventListener("popstate", restoreResult);

  const root = new URL(document.documentElement.dataset.content_root, location.href);
  const details = window.signaLearnSearchDetails || {};
  const pageNames = Object.keys(details).sort((a, b) => b.length - a.length);
  const enhanceResults = () => {
    const input = document.querySelector('#pst-search-dialog input[name="q"]');
    const terms = (input?.value || query || "").toLowerCase().split(/\s+/).filter(Boolean);
    document.querySelectorAll("#search-results .search > li").forEach((item) => {
      if (item.dataset.enhanced) return;
      const link = item.querySelector("a");
      if (!link) return;
      const url = new URL(link.getAttribute("href"), root);
      const key = pageNames.find((name) => url.pathname.endsWith(`/${name}`));
      const page = details[key];
      if (!page) return;
      item.dataset.enhanced = "true";
      const score = (entry) => terms.reduce((total, term) => total
        + (entry.title.toLowerCase().includes(term) ? 5 : 0)
        + (entry.text.toLowerCase().includes(term) ? 1 : 0), 0);
      const entry = page.entries.find((entry) => entry.anchor === decodeURIComponent(url.hash.slice(1)))
        || [...page.entries].sort((a, b) => score(b) - score(a))[0];
      if (!entry) return;
      if (!url.hash && entry.anchor) {
        link.setAttribute("href", `${document.documentElement.dataset.content_root}${key}#${entry.anchor}`);
      }
      const text = entry.text;
      const start = Math.max(0, Math.min(...terms.map((term) => text.toLowerCase().indexOf(term))
        .filter((position) => position >= 0)) - 80);
      const offset = Number.isFinite(start) ? start : 0;
      const excerpt = `${offset ? "…" : ""}${text.slice(offset, offset + 240)}${text.length > offset + 240 ? "…" : ""}`;
      const summary = document.createElement("span");
      summary.className = "context";
      const pattern = terms.length ? new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "gi") : null;
      for (const part of pattern ? excerpt.split(pattern) : [excerpt]) {
        if (terms.includes(part.toLowerCase())) {
          const mark = document.createElement("mark");
          mark.textContent = part;
          summary.append(mark);
        } else summary.append(document.createTextNode(part));
      }
      const prefix = document.documentElement.dataset.content_root;
      const breadcrumb = document.createElement("span");
      breadcrumb.className = "search-result-location";
      const addCrumb = (title, href) => {
        if (breadcrumb.childNodes.length) breadcrumb.append(document.createTextNode(" › "));
        const crumb = document.createElement("a");
        crumb.textContent = title;
        crumb.setAttribute("href", href);
        breadcrumb.append(crumb);
      };
      if (page.section && page.section !== page.title) {
        addCrumb(page.section, `${prefix}${key.split("/")[0]}/index.html`);
      }
      addCrumb(page.title, `${prefix}${key}`);
      if (entry.title !== page.title) {
        addCrumb(entry.title, link.getAttribute("href"));
      }
      const destination = new URL(link.getAttribute("href"), root);
      const searchQuery = input?.value || query || "";
      if (searchQuery.trim()) destination.searchParams.set("highlight", searchQuery);
      link.setAttribute("href", `${prefix}${key}${destination.search}${destination.hash}`);
      link.className = "search-result-target";
      link.setAttribute("aria-label", `Open ${page.title}${entry.title !== page.title ? `: ${entry.title}` : ""}`);
      link.textContent = "";
      item.replaceChildren(link, breadcrumb, document.createTextNode(" "), summary);
    });
  };
  new MutationObserver(enhanceResults).observe(document.body, {
    childList: true,
    subtree: true,
  });
  enhanceResults();
});
