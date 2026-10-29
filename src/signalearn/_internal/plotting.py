import warnings
from contextlib import contextmanager
from textwrap import fill

from dask import compute
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from .data import align_field, materialise_value, require_scalar
from .style import plotting_style
from .saving import save_destination
from .tensor import batch_task


def plot_figure(plot):
    if isinstance(plot, Figure):
        return plot
    if not isinstance(plot, tuple) or len(plot) != 2 or not isinstance(plot[0], Figure):
        raise TypeError("save_plot requires a Matplotlib Figure or a (figure, axes) plotting result")
    figure, axes = plot
    axes = np.asarray(axes, dtype=object)
    if not axes.size or any(not isinstance(ax, Axes) for ax in axes.flat):
        raise TypeError("The plotting result must contain a Matplotlib Axes or a collection of axes")
    if any(ax.figure is not figure for ax in axes.flat):
        raise ValueError("The plotting result's axes must belong to its figure")
    return figure


def plot_destination(figure, path):
    destination = save_destination(path, kind="Plot")
    format = destination.suffix[1:].lower()
    if not format:
        raise ValueError("Plot destinations require a file extension, such as .pdf, .svg, or .png")
    supported = figure.canvas.get_supported_filetypes()
    if format not in supported:
        raise ValueError(f"Unsupported plot format {format!r}; supported formats: {', '.join(sorted(supported))}")
    return destination, format


def field_text(field, fallback):
    label = field.label_latex if field.label_latex is not None else field.label
    if label is None:
        label = field.name or fallback
    unit = field.unit_latex if field.unit_latex is not None else field.unit
    return f"{label} ({unit})" if unit else label


def numeric_plot_values(fields, by, *, keep_missing=False):
    from ..data import Field

    for name, field in fields.items():
        if not isinstance(field, Field):
            raise TypeError(f"{name} must be a SignaLearn Field")
        require_scalar(field, "This plot")
        if field.dtype not in {"int64", "float64"}:
            raise TypeError(f"{name} must be a numeric field")
    anchor = next(iter(fields.values()))
    aligned = [align_field(field, anchor.ids, allow_extra=True) for field in fields.values()]
    if by is not None:
        if not isinstance(by, Field):
            raise TypeError("by must be a SignaLearn Field")
        require_scalar(by, "Plot grouping")
        aligned.append(align_field(by, anchor.ids, allow_extra=True))
    arrays = compute(*(field._values for field in aligned))
    valid = np.array([all(value is not None for value in row)
                      for row in zip(*arrays[:len(fields)])], dtype=bool)
    if not valid.any():
        raise ValueError("No observations have values for all plotted numeric fields")
    omitted = len(valid) - int(valid.sum())
    if omitted:
        names = " or ".join(fields)
        suffix = " Gaps are retained in the line." if keep_missing else ""
        warnings.warn(f"{omitted} observations omitted because {names} is missing.{suffix}",
                      UserWarning, stacklevel=4)
    selection = slice(None) if keep_missing else valid
    values = [np.asarray(array[selection], dtype=float) for array in arrays[:len(fields)]]
    return values, arrays[-1][selection] if by is not None else None


def scatter_values(x, y, by):
    values, groups = numeric_plot_values({"x": x, "y": y}, by)
    return values[0], values[1], groups


def scatter_groups(values, field):
    for category in dict.fromkeys(values):
        mask = np.array([value == category for value in values], dtype=bool)
        colour = field.colours.get(category)
        if category is None and colour is None:
            colour = "#808080"
        marker = field.markers.get(category, field.marker)
        yield mask, "Missing" if category is None else str(category), colour, marker


def validate_axes(ax):
    if ax is not None and not isinstance(ax, Axes):
        raise TypeError("ax must be a Matplotlib Axes")


@contextmanager
def plot_axes(ax):
    with plotting_style():
        if ax is None:
            from matplotlib import pyplot as plt

            figure, ax = plt.subplots(layout="constrained")
        else:
            figure = ax.figure
        yield figure, ax


def label_axes(ax, x, y):
    for field, fallback, set_label in ((x, "x", ax.set_xlabel), (y, "y", ax.set_ylabel)):
        if field is None:
            field = ""
        if not isinstance(field, str):
            field = field_text(field, fallback)
        set_label(field)


def group_legend(ax, by, handles=None, labels=None):
    if by is not None:
        ax.legend(handles=handles, labels=labels, title=field_text(by, "Group"))


def distribution_values(x, by):
    values, groups = numeric_plot_values({"x": x}, by)
    return values[0], groups


def distribution_groups(values, groups, x, by):
    if by is None:
        yield values, None, x.colour, x.marker
    else:
        for mask, label, colour, marker in scatter_groups(groups, by):
            yield values[mask], label, colour, marker


def histogram_data(values, series, bins, density):
    if isinstance(bins, (bool, np.bool_)):
        raise TypeError("bins must be a positive integer, a binning rule, or bin edges")
    edges = np.histogram_bin_edges(values, bins=bins)
    if len(edges) < 2 or not np.isfinite(edges).all() or not (np.diff(edges) > 0).all():
        raise ValueError("Histogram bin edges must be finite and strictly increasing")
    result = []
    for samples, label, colour, _ in series:
        counts, _ = np.histogram(samples, bins=edges)
        if density:
            total = counts.sum()
            if total == 0:
                raise ValueError(f"No observations fall within the density histogram bins for {label or 'x'}")
            counts = counts / (total * np.diff(edges))
        result.append((counts, label, colour))
    return edges, result


def draw_boxes(ax, series):
    artists = ax.boxplot([samples for samples, _, _, _ in series], patch_artist=True,
                         whis=1.5, notch=False, showmeans=False, showfliers=True,
                         manage_ticks=False)
    colours = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["black"])
    for index, (_, _, colour, marker) in enumerate(series):
        colour = colours[index % len(colours)] if colour is None else colour
        artists["boxes"][index].set_facecolor(colour)
        artists["boxes"][index].set_alpha(0.5)
        artists["fliers"][index].set_markeredgecolor(colour)
        artists["fliers"][index].set_markerfacecolor(colour)
        if marker is not None:
            artists["fliers"][index].set_marker(marker)
    if series[0][1] is None:
        ax.set_xticks([])
    else:
        labels = [fill(label, width=24, break_long_words=False, break_on_hyphens=False)
                  for _, label, _, _ in series]
        ax.set_xticks(range(1, len(series) + 1), labels)


def confusion_ticks(ax, labels):
    labels = [fill(str(value), width=20, break_long_words=False, break_on_hyphens=False) for value in labels]
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xticks([], minor=True)
    ax.set_yticks([], minor=True)


def annotate_confusion(ax, image, values, normalized):
    for row, column in np.ndindex(values.shape):
        value = values[row, column]
        if np.isnan(value):
            label, colour = "—", "gray"
        else:
            label = f"{value:.2f}" if normalized else str(int(value))
            rgba = image.cmap(image.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            colour = "black" if luminance > 0.5 else "white"
        ax.text(column, row, label, ha="center", va="center", color=colour)


def plot_field(field, name):
    from ..data import Field

    if not isinstance(field, Field):
        raise TypeError(f"{name} must be a SignaLearn Field")
    if not len(field):
        raise ValueError("No samples selected for plotting")


def plot_groups(by, ids):
    if by is None:
        return None
    plot_field(by, "by")
    require_scalar(by, "Plot grouping")
    return align_field(by, ids, allow_extra=True)


def line_data(x, y, by):
    if y is None:
        y, x = x, None
    plot_field(y, "y")
    if not y.dims:
        if x is None:
            raise ValueError("Scalar line plots require explicit x and y fields")
        values, groups = numeric_plot_values({"x": x, "y": y}, by, keep_missing=True)
        missing = np.isnan(values[0]) | np.isnan(values[1])
        values[1][missing] = np.nan
        return [(values[0], values[1])], x, y, groups, False
    if len(y.dims) != 1 or np.dtype(y.dtype).kind not in "iuf":
        raise TypeError("Array line plots require one numeric dimension per sample; select or transform other axes explicitly")
    dim = y.dims[0]
    if x is not None:
        plot_field(x, "x")
        x = align_field(x, y.ids, allow_extra=True)
        if x.dims != y.dims or np.dtype(x.dtype).kind not in "iuf":
            raise ValueError("Array line coordinates must have the same named dimension as y")
    else:
        x = y.coords.get(dim)
    from ..data import Field

    x_field = isinstance(x, Field)
    if x_field and any(a.shape != b.shape for a, b in zip(x._values.items, y._values.items)):
        raise ValueError("Line coordinates must match each sample's y shape")
    grouped = plot_groups(by, y.ids)
    tasks = [batch_task(y, tuple(range(len(y))))]
    if x_field:
        tasks.append(batch_task(x, tuple(range(len(x)))))
    if grouped is not None:
        tasks.append(grouped._values)
    arrays = compute(*tasks)
    series, omitted, valid_total = [], 0, 0
    for index, values in enumerate(arrays[0]):
        coordinates = arrays[1][index] if x_field else np.arange(len(values)) if x is None else x
        coordinates = np.array(coordinates, copy=True)
        values = np.asarray(values, dtype=float)
        missing = np.isnan(values)
        if coordinates.dtype.kind in "iuf":
            missing |= ~np.isfinite(coordinates)
        omitted += int(missing.sum())
        valid_total += int((~missing).sum())
        values = np.where(missing, np.nan, values)
        series.append((coordinates, values))
    if not valid_total:
        raise ValueError("No valid line coordinate pairs remain")
    if omitted:
        warnings.warn(f"{omitted} points omitted because line coordinates are missing; gaps are retained.",
                      UserWarning, stacklevel=3)
    return series, x if x_field else dim if x is not None else f"{dim} index", y, arrays[-1] if grouped is not None else None, True


def line_styles(series, groups, y, by, arrays):
    if by is None:
        for index, (x, values) in enumerate(series):
            yield x, values, y.colour, y.marker, y.ids[index] if arrays else None
        return
    for mask, label, colour, marker in scatter_groups(groups, by):
        if arrays:
            for index in np.flatnonzero(mask):
                coordinates, values = series[index]
                yield coordinates, values, colour, marker, label
        else:
            yield series[0][0][mask], series[0][1][mask], colour, marker, label


def image_axis(field, dim, size):
    from ..data import Field

    coordinate = field.coords.get(dim)
    if coordinate is None:
        return (-0.5, size - 0.5), dim
    values = materialise_value(coordinate, 0) if isinstance(coordinate, Field) else coordinate
    if values.dtype.kind not in "iuf" or len(values) < 2:
        raise ValueError(f"Image {dim} coordinates need at least two regularly spaced numeric pixel centres")
    values = np.asarray(values, dtype=float)
    differences = np.diff(values)
    monotonic = (differences > 0).all() or (differences < 0).all()
    if (not np.isfinite(values).all() or not monotonic
            or not np.allclose(differences, differences[0], rtol=1e-7, atol=abs(differences[0]) * 1e-8)):
        raise ValueError(f"Image {dim} coordinates must be finite, monotonic, and regularly spaced; no resampling is inferred")
    step = differences[0]
    return (values[0] - step / 2, values[-1] + step / 2), coordinate if isinstance(coordinate, Field) else dim


def image_data(x):
    plot_field(x, "x")
    if len(x) != 1:
        raise ValueError("plot_image requires one sample; select an image with data[index].image")
    if set(x.dims) not in ({"height", "width"}, {"height", "width", "channel"}):
        raise ValueError("Images require height and width dimensions, with an optional channel dimension")
    order = tuple(x.dims.index(dim) for dim in ("height", "width", "channel") if dim in x.dims)
    shape = x._values.items[0].shape
    if any(size == 0 for size in shape):
        raise ValueError("Cannot plot an image with empty dimensions")
    if "channel" in x.dims and shape[x.dims.index("channel")] not in {3, 4}:
        raise ValueError("Colour images require three RGB or four RGBA channels")
    horizontal, x_label = image_axis(x, "width", shape[x.dims.index("width")])
    vertical, y_label = image_axis(x, "height", shape[x.dims.index("height")])
    values = np.transpose(materialise_value(x, 0), order)
    if values.ndim == 3:
        channels = x.coords.get("channel")
        if channels is not None:
            from ..data import Field

            channels = materialise_value(channels, 0) if isinstance(channels, Field) else channels
            names = [str(value).lower() for value in channels]
            expected = ["red", "green", "blue", "alpha"][:values.shape[2]]
            short = ["r", "g", "b", "a"][:values.shape[2]]
            if names != expected and names != short:
                raise ValueError("Colour channel coordinates must declare RGB or RGBA order")
        if values.dtype != np.uint8 and (values.dtype.kind != "f" or not np.isfinite(values).all()
                                        or values.min() < 0 or values.max() > 1):
            raise ValueError("RGB/RGBA images require uint8 values or finite floats in [0, 1]; scale explicitly")
    elif values.dtype.kind == "f":
        if np.isnan(values).all():
            raise ValueError("No valid pixels remain")
        omitted = int(np.isnan(values).sum())
        if omitted:
            warnings.warn(f"{omitted} missing pixels masked in the image.", UserWarning, stacklevel=3)
        values = np.ma.masked_invalid(values)
    return values, (horizontal[0], horizontal[1], vertical[1], vertical[0]), x_label, y_label
