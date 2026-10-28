from pathlib import Path
import tomllib

project = "SignaLearn"
author = "Kyriacos Kyriacou"
release = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())["project"]["version"]
extensions = ["myst_parser"]
source_suffix = {".md": "markdown"}
root_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
myst_heading_anchors = 4
myst_enable_extensions = ["colon_fence", "dollarmath"]
nitpicky = True
html_theme = "pydata_sphinx_theme"
html_title = "SignaLearn"
html_baseurl = "https://kyri04.github.io/signalearn/"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["navigation.css"]
html_js_files = ["search-data.js", "search.js", "navigation.js"]
html_show_search_summary = False
html_context = {"default_mode": "auto"}
html_sidebars = {"**": ["sidebar-tree.html"]}
html_theme_options = {
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": ["search-button-field.html"],
    "search_as_you_type": True,
    "search_bar_text": "Search documentation",
    "navigation_depth": 3,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "secondary_sidebar_items": [],
    "article_header_start": [],
    "show_prev_next": False,
    "back_to_top_button": False,
    "footer_start": [],
    "footer_end": [],
}
html_show_sourcelink = False
html_permalinks = False


def write_search_details(app, exception):
    if exception or app.builder.format != "html":
        return
    import json
    from docutils import nodes

    pages = {}
    sections = {"guide": "Guide", "reference": "Reference", "extensions": "Extensions"}
    for name in sorted(app.env.found_docs):
        tree = app.env.get_doctree(name)
        title = app.env.titles[name].astext()
        entries = []
        for section in tree.findall(nodes.section):
            heading = next(iter(section.findall(nodes.title)), None)
            entries.append({
                "anchor": section["ids"][0] if section["ids"] else "",
                "title": heading.astext() if heading is not None else title,
                "text": " ".join(section.astext().split()),
            })
        pages[app.builder.get_target_uri(name)] = {
            "title": title,
            "section": sections.get(name.split("/")[0], ""),
            "entries": entries,
        }
    target = Path(app.outdir) / "_static" / "search-data.js"
    target.write_text("window.signaLearnSearchDetails = " + json.dumps(pages, ensure_ascii=False) + ";\n")


def setup(app):
    app.connect("build-finished", write_search_details)
