from importlib.resources import as_file, files
from pathlib import Path

import matplotlib as mpl
from matplotlib import style


_style = {}


def configure_plot_style(path):
    global _style
    if path is None or isinstance(path, str) and path == "academic":
        resource = files("signalearn").joinpath("styles", "academic.mplstyle")
        with as_file(resource) as bundled_path:
            settings = mpl.rc_params_from_file(
                bundled_path, fail_on_error=True, use_default_template=False,
            )
    else:
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Plot style file not found: {path}; built-in style: academic")
        settings = mpl.rc_params_from_file(path, fail_on_error=True, use_default_template=False)
    if settings.get("text.usetex", False):
        raise ValueError("External LaTeX rendering is not supported; use Mathtext with text.usetex: False")
    _style = dict(settings)


def plotting_style():
    return style.context(_style, after_reset=True)


configure_plot_style(None)
