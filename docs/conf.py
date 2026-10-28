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
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["navigation.css"]
html_context = {"default_mode": "auto"}
html_sidebars = {"**": ["sidebar-tree.html"]}
html_theme_options = {
    "navbar_center": [],
    "navbar_end": [],
    "navbar_persistent": ["search-field.html"],
    "search_bar_text": "Search documentation",
    "navigation_depth": 3,
    "show_nav_level": 2,
    "collapse_navigation": False,
    "secondary_sidebar_items": [],
    "article_header_start": [],
    "show_prev_next": False,
    "footer_start": [],
    "footer_end": [],
}
html_show_sourcelink = False
