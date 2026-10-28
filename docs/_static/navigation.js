document.addEventListener("DOMContentLoaded", () => {
  const scroller = document.querySelector(".sl-navigation-scroll");
  const dialog = document.getElementById("pst-primary-sidebar-modal");
  const toggle = document.querySelector(".primary-toggle");
  if (!scroller || !dialog || !toggle) return;
  const wide = window.matchMedia("(min-width: 960px)");
  const navigation = scroller.querySelector(".bd-docs-nav");
  const updateFades = () => {
    const remaining = scroller.scrollHeight - scroller.clientHeight - scroller.scrollTop;
    const above = scroller.scrollTop > 1;
    const below = remaining > 1;
    scroller.style.setProperty("--sl-sidebar-clear-top", above ? "24px" : "0px");
    scroller.style.setProperty("--sl-sidebar-fade-top", above ? "64px" : "0px");
    scroller.style.setProperty("--sl-sidebar-clear-bottom", below ? "12px" : "0px");
    scroller.style.setProperty("--sl-sidebar-fade-bottom", below ? "48px" : "0px");
  };
  let savedScroll = 0;
  try { savedScroll = Number(sessionStorage.getItem("sl-navigation-scroll")) || 0; } catch {}
  scroller.scrollTop = savedScroll;
  scroller.addEventListener("scroll", () => {
    savedScroll = scroller.scrollTop;
    try { sessionStorage.setItem("sl-navigation-scroll", savedScroll); } catch {}
    updateFades();
  }, { passive: true });
  const observer = new ResizeObserver(updateFades);
  observer.observe(scroller);
  if (navigation) observer.observe(navigation);
  dialog.setAttribute("aria-labelledby", "sl-navigation-title");
  document.querySelector(".sl-navigation-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (event) => {
    if (event.button === 0 && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey
      && event.target.closest(".bd-docs-nav a, .sl-navigation-header a")) dialog.close();
  });
  new MutationObserver(() => {
    if (dialog.open) dialog.focus({ preventScroll: true });
    toggle.setAttribute("aria-expanded", String(dialog.open));
    toggle.setAttribute("aria-label", dialog.open ? "Close navigation" : "Open navigation");
    requestAnimationFrame(() => {
      scroller.scrollTop = savedScroll;
      updateFades();
    });
  }).observe(dialog, { attributes: true, attributeFilter: ["open"] });
  wide.addEventListener("change", () => {
    if (wide.matches && dialog.open) dialog.close();
    requestAnimationFrame(updateFades);
  });
  window.addEventListener("pageshow", updateFades);
  updateFades();
});
