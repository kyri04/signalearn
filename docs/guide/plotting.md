# Plotting
Plots use field metadata for labels, units, colours, and markers. Every plotting
function returns `(figure, axes)` and accepts existing axes.

```python
import signalearn as sl

fig, ax = sl.plot_hist(data.temperature, by=data.state)
ax.set_title("Temperature by state")
```

## Plots
| Plot | Function |
| --- | --- |
| Scatter | `plot_scatter` |
| Histogram | `plot_hist` |
| Box | `plot_box` |
| Image | `plot_image` |
| Line | `plot_line` |
| Confusion matrix | `plot_confusion_matrix` |

Pass `by` for grouping where supported. Category colours come from
`by.colours`; ungrouped scatter and line plots use the y field's appearance.
Set category colours explicitly for consistency across subsets.

Missing scalar measurements are omitted with a warning; line plots show gaps.
The dataset is unchanged. Image plotting validates pixel values and axes
without silently rescaling them.

## Styles
```python
sl.set_plot_style("academic")
```

The built-in style uses STIX fonts and Mathtext, with no external LaTeX
installation. Load a custom Matplotlib style with its path:

```python
sl.set_plot_style("styles/custom.mplstyle")
```

To start from the bundled file, copy `src/signalearn/styles/academic.mplstyle`
from the checkout or read it with `importlib.resources.files("signalearn")`.
Each selection starts from Matplotlib defaults. Styles do not change the global
backend or settings; reselect an edited file to reload it.

## Layout
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(8, 3))
sl.plot_hist(data.temperature, by=data.state, ax=axes[0])
sl.plot_box(data.temperature, by=data.state, ax=axes[1])
fig.tight_layout()
```

Use returned Matplotlib objects for individual titles, limits, and figure size.
Existing axes keep their appearance settings. Style-file sizes are in inches;
`fig.set_size_inches(fig.get_size_inches() * 0.8)` resizes proportionally.

Display follows your backend; SignaLearn does not call `show`, save, or close
figures. See [Saving](saving.md) for export and the
[plotting reference](../reference/plotting.md) for styles and plot-specific rules.
