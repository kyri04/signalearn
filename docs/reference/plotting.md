# Plotting
Defined in `signalearn.plotting`. Plots use Matplotlib, return `(Figure, Axes)`,
accept existing axes without clearing artists, and do not show/save/close them.
Styles are scoped, preserving global rc settings and the active backend.

## Inputs
Scalar coordinates are numeric Fields, not raw arrays or strings. The first
coordinate Field defines the plotted samples; other coordinates and `by` must
cover its IDs. Extra supporting IDs are ignored. Array line plots use the signal's
IDs. See [alignment](data.md#alignment). Only selected
Fields/dependencies are computed together. Category grouping is scalar and
supports strings, booleans, integers, or floats. SampleFields are accepted where
their shape is appropriate.

Axis/legend text prefers label_latex, then label, then field name; units prefer
unit_latex, then unit, and appear in parentheses. Unnamed coordinates use x/y.
Mathtext formatting does not change values/units. Invalid expressions fail on
rendering. Group appearance uses by.colours/by.markers; ungrouped series use
their Field's colour/marker. Unconfigured categories use the active colour cycle;
group markers fall back to by.marker then Matplotlib's default. Missing groups
are grey and labelled Missing. Explicit palettes remain consistent across subsets.

## plot_scatter
```python
sl.plot_scatter(x, y, by=None, ax=None)
```

Plot scalar numeric pairs. Ungrouped appearance comes from y. Omit missing x/y
rows with a warning, retaining missing groups when coordinates exist. No valid
pairs raises ValueError; the Dataset remains unchanged. Marker area is
lines.markersize squared (points squared); opacity is fixed at 0.8 because
Matplotlib has no general scatter-opacity style setting. Customise the returned
collection for a one-off change.

## plot_hist
```python
sl.plot_hist(x, by=None, bins="fd", density=False, ax=None)
```

Overlay scalar distributions, grouped in order of first valid appearance.
Bins accept a positive integer, supported NumPy rule, or strictly increasing
finite edges. Default fd is Freedman–Diaconis. Calculate common edges across
all valid values; explicit edges exclude values outside them from the display.
Histograms are filled with opacity 0.5, overlaid rather than stacked.

Default heights are counts. Density independently normalises each group to area
one over selected bins, in inverse x units; heights are not individual-bin
probabilities. A group with no values inside bins is an error for density and
zero heights for counts. Missing x is omitted with warning. Empty/all-missing x
is an error. Ungrouped colour comes from x.

## plot_box
```python
sl.plot_box(x, by=None, ax=None)
```

One vertical box per observed group, or one ungrouped box, in first-valid order.
Boxes span quartiles, show median, and use whiskers to extreme values within
1.5 IQR. Outliers remain visible; no notches or means are drawn. Matplotlib's
quartile calculation applies. Y uses x's label/unit; grouped x labels wrap.

Fill opacity is 0.5. Colours/markers use x ungrouped or by category mappings.
Outlier style otherwise follows the selected style. Missing x is omitted with
warning; groups without valid x have no box. Empty/all-missing x is an error.

## plot_image
```python
sl.plot_image(x, ax=None)
```

Require exactly one selected image, with named height/width and optional channel
axes in any storage order. Only this sample/dependencies are read. Colour images
have three RGB or four RGBA channels. If channel coordinates exist, use red/green/
blue[/alpha] or r/g/b[/a], in order. Other meanings require explicit conversion.

Grayscale uses the active colormap/native normalisation. RGB/RGBA requires uint8
or finite floating pixels in [0,1]; other ranges/types error rather than clip.
Missing grayscale pixels are masked with warning; all-missing images error.
Colour pixels must be complete. Grayscale has no automatic colour bar.

Without spatial coordinates, centres start at zero. Supplied numeric centres
must be finite, monotonic, regularly spaced, and contain at least two values.
Coordinate Fields supply axis labels/units. First height row appears at the top;
the title comes from image label/unit. No interpolation or resampling is inferred
for irregular coordinates. Image cmap/interpolation belong in the style or on
ax.images[0]. There is no gallery, video playback, channel selector, or file decoder.

## plot_line
```python
sl.plot_line(x, y=None, by=None, ax=None)
```

Two scalar numeric Fields form points connected in current sample order,
separately per group. One-dimensional array Fields form one line per selected
sample, allowing ragged lengths. With only one array argument, it supplies y
and its coordinate supplies x; otherwise x uses indices with an explicit
dimension-index label. Explicit array x/y need identical dimensions and per-sample
shapes. Higher ranks require selection/transform first.

Inputs/groups align by ID. No sorting/interpolation or alignment by dimension
name alone occurs. Inline coordinate labels use the dimension name; referenced
Fields supply labels/units. Missing points produce gaps and warnings rather
than connecting across them. No valid pairs errors. Ungrouped array artists
identify sample IDs without an automatic legend; grouped lines have one legend
entry per category. Width/default markers use the style.

## plot_confusion_matrix
```python
sl.plot_confusion_matrix(predictions, target, normalize=None, ax=None)
```

Use [metric input alignment](evaluation.md#inputs) with ordinary
class labels. Rows are true classes, columns predicted classes, both sorted by
the observed union. Annotated counts include every aligned sample. Normalize
None for counts, true by row, pred by column, or all by total. Zero-denominator
rows/columns warn and are masked with an em dash. Defined proportions display
two decimals, without rounding stored values. A colour bar indicates the scale.
Formatted target labels and image.cmap are used. Access the matrix/artist through
ax.images[0].

## set_plot_style
```python
sl.set_plot_style(path=None)
```

Select the built-in academic style with None/no arguments or the exact string
academic; otherwise supply a local mplstyle string/Path. Path("academic") is
a literal file path. Relative paths follow the working directory and ~ expands.
URLs and other built-in Matplotlib style names are unsupported.

Each selection starts from Matplotlib defaults, rather than inheriting a previous
style or global user formatting. Reselect a file after editing. Missing/invalid
settings error without replacing the active style; unknown keys/malformed lines
use Matplotlib warnings. Styles enabling text.usetex are rejected; Mathtext works
without external LaTeX. Existing figures remain unchanged. New sessions must
select their custom style again.

Academic uses STIXGeneral/STIX, inward ticks, no grid, 5x3.5 inches at display
100 DPI; savefig uses 300 DPI, tight bounds, and 0.05-inch padding. Fonts ship
with Matplotlib. Figure sizes in styles are absolute inches. Existing axes keep
their fonts/grid; new artists use supported shared settings. Backend settings
are ignored by Matplotlib style loading. See [custom styles](../guide/plotting.md#styles).

## save_plot
```python
sl.save_plot(plot, path)
```

Accept a Figure or two-item (Figure, Axes/collection) tuple. Collections must
be non-empty and all axes must belong to that Figure. A standalone Axes is not
accepted. Export the entire figure, including panels, legends, titles, and bars.
Closed/native Matplotlib figures may also be saved without reopening.

Require an explicit extension supported by the figure's canvas, case-insensitively,
retaining the supplied filename. Unsupported/missing extensions fail before
creating directories. Return an absolute Path; create parents, expand ~, and
resolve relative paths from the working directory. Reject files/directories/
symlinks at existing destinations. Exclusive creation also prevents overwriting
a concurrently created destination. Failed writes remove the partial owned file.

Use the style selected at save time for export settings, retaining current
artists/customisations. Raster DPI and rasterised vector elements follow
savefig.dpi. Tight bounds/padding alter physical export size; savefig.bbox:
standard keeps the canvas. Format always comes from the extension. Save leaves
the figure open and preserves display DPI, canvas, backend, and global settings.
Normal Matplotlib rendering/layout still applies. It does not rerun plotting or
read Fields again. Output is an image/vector file without a Python sidecar.
Use native Figure.savefig for individual options outside shared styles.
