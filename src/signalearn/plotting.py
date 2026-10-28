from collections.abc import Sequence
from os import PathLike
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ._internal.evaluation import classification_data, confusion_values
from ._internal.plotting import (
    annotate_confusion, confusion_ticks, distribution_groups, distribution_values, draw_boxes, field_text,
    group_legend, histogram_data, image_data, label_axes, line_data, line_styles,
    plot_axes, plot_destination, plot_figure, scatter_groups, scatter_values,
    validate_axes,
)
from ._internal.saving import output_file
from ._internal.style import configure_plot_style, plotting_style
from .data import Dataset, Field

__all__ = ["plot_scatter", "plot_hist", "plot_box", "plot_image", "plot_line", "plot_confusion_matrix", "set_plot_style",
           "save_plot"]


def set_plot_style(path: str | PathLike[str] | None = None) -> None:
    configure_plot_style(path)


def save_plot(plot: Figure | tuple, path: str | PathLike[str]) -> Path:
    figure = plot_figure(plot)
    destination, format = plot_destination(figure, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with plotting_style(), output_file(destination) as stream:
        figure.savefig(stream, format=format)
    return destination


def plot_scatter(
    x: Field, y: Field, by: Field | None = None, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    validate_axes(ax)
    x_values, y_values, groups = scatter_values(x, y, by)
    with plot_axes(ax) as (figure, ax):
        if by is None:
            ax.scatter(x_values, y_values, color=y.colour, marker=y.marker, alpha=0.8)
        else:
            handles, labels = [], []
            for mask, label, colour, marker in scatter_groups(groups, by):
                handles.append(ax.scatter(x_values[mask], y_values[mask], color=colour,
                                          marker=marker, alpha=0.8, label=label))
                labels.append(label)
            group_legend(ax, by, handles, labels)
        label_axes(ax, x, y)
    return figure, ax


def plot_hist(
    x: Field, by: Field | None = None, bins: int | str | Sequence[float] = "fd",
    density: bool = False, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    validate_axes(ax)
    if not isinstance(density, bool):
        raise TypeError("density must be a boolean")
    values, groups = distribution_values(x, by)
    series = list(distribution_groups(values, groups, x, by))
    edges, histograms = histogram_data(values, series, bins, density)
    with plot_axes(ax) as (figure, ax):
        for counts, label, colour in histograms:
            ax.stairs(counts, edges, fill=True, color=colour, alpha=0.5, label=label)
        group_legend(ax, by)
        label_axes(ax, x, "Density" if density else "Count")
    return figure, ax


def plot_box(
    x: Field, by: Field | None = None, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    validate_axes(ax)
    values, groups = distribution_values(x, by)
    series = list(distribution_groups(values, groups, x, by))
    with plot_axes(ax) as (figure, ax):
        draw_boxes(ax, series)
        label_axes(ax, field_text(by, "Group") if by is not None else None, x)
    return figure, ax


def plot_image(x: Field, ax: Axes | None = None) -> tuple[Figure, Axes]:
    validate_axes(ax)
    values, extent, x_label, y_label = image_data(x)
    with plot_axes(ax) as (figure, ax):
        ax.imshow(values, extent=extent, origin="upper")
        label_axes(ax, x_label, y_label)
        ax.set_title(field_text(x, "Image"))
    return figure, ax


def plot_confusion_matrix(
    predictions: Field | Dataset, target: Field, normalize: str | None = None, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    validate_axes(ax)
    if normalize is not None and (not isinstance(normalize, str) or normalize not in {"true", "pred", "all"}):
        raise ValueError("normalize must be 'true', 'pred', 'all', or None")
    data = classification_data(predictions, target)
    values = confusion_values(data, normalize)
    with plot_axes(ax) as (figure, ax):
        image = ax.imshow(values, origin="upper", vmin=0, vmax=1 if normalize is not None else None)
        confusion_ticks(ax, data.labels)
        annotate_confusion(ax, image, values, normalize is not None)
        label = field_text(data.target, "Class")
        label_axes(ax, f"Predicted {label}", f"True {label}")
        figure.colorbar(image, ax=ax, label="Proportion" if normalize is not None else "Count")
    return figure, ax


def plot_line(
    x: Field, y: Field | None = None, by: Field | None = None, ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    validate_axes(ax)
    series, x_label, y_field, groups, arrays = line_data(x, y, by)
    with plot_axes(ax) as (figure, ax):
        handles, labels = [], []
        colours = {}
        for coordinates, values, colour, marker, label in line_styles(series, groups, y_field, by, arrays):
            if by is not None and colour is None:
                colour = colours.get(label)
            line, = ax.plot(coordinates, values, color=colour, marker=marker, label=label)
            if by is not None and label not in labels:
                handles.append(line)
                labels.append(label)
                colours[label] = line.get_color()
        group_legend(ax, by, handles, labels)
        label_axes(ax, x_label, y_field)
    return figure, ax
