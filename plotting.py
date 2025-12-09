import matplotlib.pyplot as plt
import numpy as np
from signalearn.utility import *
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    fig.savefig(f"{filename}.{extension}", bbox_inches='tight', dpi=dpi)

def _select_feature_importances(res, feature_attr):
    feature_map = getattr(res.meta, "feature_importances", None)
    if not isinstance(feature_map, dict) or not feature_map:
        legacy = getattr(res.results, "feature_importances", None)
        if isinstance(legacy, dict) and legacy:
            feature_map = legacy
        else:
            return None, None

    if feature_attr not in feature_map:
        raise ValueError(f"Feature attribute '{feature_attr}' not found in result importances.")

    values = np.asarray(feature_map[feature_attr], dtype=float)
    return feature_attr, values

def plot_importances(
    result,
    points,
    x_attr,
    y_attr,
    func=None,
    yfunc=np.mean,
):
    if not points:
        raise ValueError("plot_importances requires a non-empty 'points' collection.")
    if y_attr is None:
        raise ValueError("y_attr must be provided to match feature importances.")

    plt.close('all')
    fig, ax = plt.subplots()
    xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)
    identity = (lambda x: x)
    transform = func if callable(func) else identity

    if isinstance(result, list):
        importances = []
        first = None
        for r in result:
            _, fi = _select_feature_importances(r, y_attr)
            if fi is not None:
                importances.append(np.asarray(fi, dtype=float))
                if first is None:
                    first = r

        if not importances or first is None:
            raise ValueError("No feature importances available for plotting.")

        imp = np.mean(importances, axis=0)
        first_res = first
        if x_attr is None:
            raise ValueError("x_attr must be provided when plotting averaged results.")
        x_imp = np.asarray(getattr(first_res.params.mean_point, x_attr), dtype=float)
    else:
        _, fi = _select_feature_importances(result, y_attr)
        if fi is None:
            raise ValueError("No feature importances available for plotting.")
        imp = np.asarray(fi, dtype=float)

        if x_attr is None:
            x_imp = np.asarray(getattr(points[0], y_attr), dtype=float)
        else:
            x_imp = np.asarray(getattr(points[0], x_attr), dtype=float)

    ax.plot(x_imp, imp, label="Feature importances")

    if points:
        Ys = []
        for p in points:
            Ys.append(transform(getattr(p, y_attr)))

        Y = np.vstack(Ys)
        try:
            ref_curve = yfunc(Y, axis=0)
        except TypeError:
            ref_curve = yfunc(Y)

        imp_min, imp_max = float(np.min(imp)), float(np.max(imp))
        med_min, med_max = float(np.min(ref_curve)), float(np.max(ref_curve))
        if med_max > med_min:
            scale = (imp_max - imp_min) / (med_max - med_min) if imp_max > imp_min else 1.0
            offset = imp_min - scale * med_min
            med_scaled = scale * ref_curve + offset
        else:
            med_scaled = np.full_like(ref_curve, imp_min)

        func_name = pretty(yfunc.__name__) if callable(yfunc) else ""
        lbl = f"{func_name} {ylabel}" if func_name else ylabel
        ax.plot(x_imp, med_scaled, linestyle=":", color="0", label=lbl)

    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Importance")
    if points:
        ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_scatter(points, x_attr, y_attr, group_attr=None):
    plt.close('all')
    fig, ax = plt.subplots()
    if group_attr is None:
        groups = {None: points}
    else:
        groups = {}
        for p in points:
            g = getattr(p, group_attr, None)
            groups.setdefault(g, []).append(p)
    for i, (g, pts) in enumerate(groups.items()):
        xs = [getattr(p, x_attr) for p in pts]
        ys = [getattr(p, y_attr) for p in pts]
        lbl = str(g) if group_attr is not None else None
        ax.scatter(xs, ys, alpha=0.7, edgecolors='k', label=lbl)
    xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if group_attr is not None:
        ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_point(point, x_attr, y_attr, func=None):
    plt.close('all')
    fig, ax = plt.subplots()

    X = getattr(point, x_attr)
    Y = getattr(point, y_attr)

    if func is not None:
        Y_plot = func(Y)
    else:
        Y_plot = Y

    xlabel, ylabel = get_axes_labels(point, x_attr, y_attr)

    ax.plot(X, Y_plot, lw=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax

def plot_point_peaks(point, x_attr, y_attr, pos_attr, fwhm_attr, func=None, colors=None):
    plt.close('all')
    fig, ax = plt.subplots()

    X = np.asarray(getattr(point, x_attr), dtype=float)
    Y_raw = getattr(point, y_attr)
    Y = np.asarray(Y_raw, dtype=float)

    if func is not None:
        Y_plot = func(Y)
    else:
        Y_plot = Y

    xlabel, ylabel = get_axes_labels(point, x_attr, y_attr)
    ax.plot(X, Y_plot, lw=1, color="C0")

    if isinstance(pos_attr, str):
        pos_attrs = [pos_attr]
    else:
        pos_attrs = list(pos_attr)

    if isinstance(fwhm_attr, str):
        fwhm_attrs = [fwhm_attr]
    else:
        fwhm_attrs = list(fwhm_attr)

    if len(pos_attrs) != len(fwhm_attrs):
        raise ValueError("pos_attr and fwhm_attr must have the same length when provided as lists.")

    if colors is None:
        color_cycle = plt.rcParams.get("axes.prop_cycle", None)
        if color_cycle is not None:
            colors = color_cycle.by_key().get("color", [])
        if not colors:
            colors = ["C1", "C2", "C3", "C4", "C5"]

    for idx_attr, (p_attr, w_attr) in enumerate(zip(pos_attrs, fwhm_attrs)):
        if not hasattr(point, p_attr) or not hasattr(point, w_attr):
            continue

        pos_vals = np.asarray(getattr(point, p_attr), dtype=float)
        fwhm_vals = np.asarray(getattr(point, w_attr), dtype=float)

        if pos_vals.shape != fwhm_vals.shape:
            raise ValueError(f"Shape mismatch between {p_attr} and {w_attr}.")

        if pos_vals.ndim == 0:
            pos_vals = pos_vals.reshape(1)
            fwhm_vals = fwhm_vals.reshape(1)

        color = colors[idx_attr % len(colors)]

        for pos, width in zip(pos_vals.ravel(), fwhm_vals.ravel()):
            if not np.isfinite(pos) or not np.isfinite(width) or width <= 0:
                continue

            # Find approximate peak height from the (possibly transformed) signature
            i = int(np.argmin(np.abs(X - pos)))
            peak_y = float(Y_plot[i]) if np.isfinite(Y_plot[i]) else np.nan
            if not np.isfinite(peak_y):
                continue

            half_width = 0.5 * width
            x_left = pos - half_width
            x_right = pos + half_width

            # Clip to the data range
            x_min, x_max = np.nanmin(X), np.nanmax(X)
            x_left = max(x_left, x_min)
            x_right = min(x_right, x_max)
            if x_right <= x_left:
                continue

            ax.axvline(pos, color=color, linestyle="--", linewidth=1, alpha=0.7)
            ax.hlines(peak_y, x_left, x_right, color=color, linewidth=2, alpha=0.9)
            ax.plot(pos, peak_y, marker="o", color=color, markersize=4)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax

def plot_points_range(points, x_attr, y_attr, func=None, q=(0.25, 0.5, 0.75), group_attr=None):
    plt.close('all')
    fig, ax = plt.subplots()

    if group_attr is None:
        X = np.asarray(getattr(points[0], x_attr))
        Y = np.vstack([(getattr(p, y_attr) if func is None else func(getattr(p, y_attr))) for p in points])

        xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)

        Y[~np.isfinite(Y)] = np.nan
        q1, med, q3 = np.nanquantile(Y, q, axis=0)

        m = np.isfinite(q1) & np.isfinite(med) & np.isfinite(q3)
        ax.fill_between(X[m], q1[m], q3[m], alpha=0.25, label='IQR')
        ax.plot(X[m], med[m], lw=1.5, label='Median')

    else:
        groups = group_points(points, group_attr)
        xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)
        for pts in groups:
            if not pts:
                continue
            X = np.asarray(getattr(pts[0], x_attr))
            Y = np.vstack([(getattr(p, y_attr) if func is None else func(getattr(p, y_attr))) for p in pts])
            Y[~np.isfinite(Y)] = np.nan
            mean = np.nanmean(Y, axis=0)
            m = np.isfinite(mean)
            gval = getattr(pts[0], group_attr, None)
            lbl = str(gval) if gval is not None else None
            ax.plot(X[m], mean[m], lw=1.5, label=lbl)

    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ylabel = pretty_func(ylabel, func)
    ax.set_ylabel(ylabel)

    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_points(points, x_attr, y_attr, func=None, offset=0.1):
    plt.close('all')
    fig, ax = plt.subplots()
    for i, p in enumerate(points):
        y_vals = getattr(p, y_attr)
        y = y_vals if func is None else func(y_vals)
        if offset:
            y = y + i * offset
        ax.plot(getattr(p, x_attr), y, alpha=0.7)

    ax.set_yticks([])

    xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)
    ylabel = pretty_func(ylabel, func)
    ax.set_ylabel(ylabel)

    ax.set_xlabel(xlabel)

    ax.margins(x=0)

    fig.tight_layout()
    return fig, ax

def plot_func(points, x_attr, y_attr, func=np.mean):
    plt.close('all')

    Y_stack = np.vstack([getattr(p, y_attr) for p in points])
    y = func(Y_stack, axis=0)

    fig, ax = plt.subplots()

    xlabel, ylabel = get_axes_labels(points[0], x_attr, y_attr)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(getattr(points[0], x_attr), y)

    fig.tight_layout()
    return fig, ax

def plot_func_difference(points_a, points_b, x_attr, y_attr, func=np.mean):
    plt.close('all')
    fig, ax = plt.subplots()

    Y_a = np.vstack([getattr(p, y_attr) for p in points_a])
    Y_b = np.vstack([getattr(p, y_attr) for p in points_b])
    y_a = func(Y_a, axis=0)
    y_b = func(Y_b, axis=0)
    y = y_a - y_b

    xlabel, ylabel = get_axes_labels(points_a[0], x_attr, y_attr)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.plot(getattr(points_a[0], x_attr), y)

    fig.tight_layout()
    return plt

def plot_distribution(points, y_attr, func=None, group_attr=None):
    plt.close('all')
    fig, ax = plt.subplots()
    groups = group_points(points, group_attr)

    def values_from(group):
        vals = []
        for p in group:
            arr = getattr(p, y_attr, None)
            if arr is None:
                continue
            vals.append(arr)
        vals = np.asarray(vals, float)
        return func(vals) if callable(func) else vals

    group_values = [values_from(g) for g in groups]
    values = np.concatenate(group_values) if group_values else np.array([])
    n = max(5, int(np.sqrt(len(values)))) if len(values) > 0 else 5

    values = values[np.isfinite(values)]
    counts, bins = np.histogram(values, bins=n)

    ax.hist(values, bins=bins, alpha=0.35, label="All", color="tab:gray")

    for i, gv in enumerate(group_values, 1):
        if len(gv) == 0:
            continue
        ax.hist(gv, bins=bins, histtype="step", linewidth=1.8, label=f"Group {i}")

    first_point = groups[0][0] if groups and groups[0] else None
    xlabel, = get_axes_labels(first_point, y_attr)
    xlabel = pretty_func(xlabel, func)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frequency")
    if len(groups) > 1:
        ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_rocs(results):
    plt.close('all')
    fig, ax = plt.subplots()
    valid = [r.meta for r in results
             if (r.meta.y_true is not None and r.meta.y_score is not None)]

    roc_data = []
    for r in valid:
        fpr, tpr, _ = roc_curve(r.y_true, r.y_score)
        auc = roc_auc_score(r.y_true, r.y_score)
        roc_data.append((auc, fpr, tpr))

    roc_data.sort(key=lambda x: x[0])
    worst_auc, worst_fpr, worst_tpr = roc_data[0]
    best_auc,  best_fpr,  best_tpr  = roc_data[-1]

    mean_grid = np.linspace(0.0, 1.0, 1001)
    tprs_interp = []
    aucs = []
    for auc, fpr, tpr in roc_data:
        tpr_interp = np.interp(mean_grid, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs_interp.append(tpr_interp)
        aucs.append(auc)
    mean_tpr = np.mean(tprs_interp, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.trapz(mean_tpr, mean_grid)

    for auc, fpr, tpr in roc_data:
        ax.plot(fpr, tpr, linewidth=0.8, alpha=0.2, color=(0.7, 0.7, 0.7))

    ax.plot([0, 1], [0, 1], linestyle=':', label=f"Baseline AUC = 0.500",
            linewidth=1.0, color=(0.5, 0.5, 0.5))
    ax.plot(best_fpr,  best_tpr,  linewidth=1.6, label=f"Best AUC = {best_auc:.3f}",  color='green')
    ax.plot(worst_fpr, worst_tpr, linewidth=1.6, label=f"Worst AUC = {worst_auc:.3f}", color='red')
    ax.plot(mean_grid, mean_tpr, linewidth=1.8, label=f"Mean AUC = {mean_auc:.3f}", color='black')

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax

def plot_learning_curve(results, volume_attr='points', result_attr='f1'):
    plt.close('all')
    fig, ax = plt.subplots()
    xs, ys = [], []
    for r in results:
        x = getattr(r.volume, volume_attr, None)
        y = getattr(r.results, result_attr, None)

        xs.append(x); ys.append(y)

    xlabel = f"Number of {volume_attr}"
    ylabel = result_attr.replace('_', ' ').capitalize()

    ax.scatter(xs, ys, s=12)
    ax.plot(xs, fit_spline(xs, ys)(xs), linestyle="--", color="black")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    return fig, ax

def plot_confusion_matrix(result):
    plt.close('all')
    fig, ax = plt.subplots()

    conf_matrix = getattr(result.meta, 'conf_matrix', None)
    labels = getattr(result.params, 'unique_labels', None)

    cm = np.asarray(conf_matrix, dtype=float)
    if labels is None or len(labels) != cm.shape[0]:
        labels = [str(i) for i in range(cm.shape[0])]
    else:
        labels = [str(lbl) for lbl in labels]

    fmt = ".0f"
    im = ax.imshow(cm, interpolation='nearest', cmap='Greys')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    label_name = getattr(result.params, 'target', None) or getattr(result.params, 'target_label', None)
    if label_name:
        x_label = f"Predicted {label_name}"
        y_label = f"True {label_name}"
    else:
        x_label = "Predicted label"
        y_label = "True label"

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title("Confusion Matrix")
    ax.grid(False)
    ax.grid(False, which='minor')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            rgba = im.cmap(im.norm(value))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            ax.text(
                j, i, format(value, fmt),
                ha='center', va='center',
                color='white' if luminance < 0.5 else 'black'
            )

    fig.tight_layout()
    return fig, ax

def plot_grid(points, x_attr, y_attr, grid_size, group_attr=None):
    plt.close('all')
    fig, ax = plt.subplots()

    groups = group_points(points, group_attr)

    cells = {}

    x_max = grid_size[0]
    y_max = grid_size[1]

    for gi, group in enumerate(groups):
        for p in group:
            x = int(getattr(p, x_attr)); y = int(getattr(p, y_attr))
            cells.setdefault((y, x), []).append(gi)

    class_names = [f"Group {i+1}" for i in range(len(groups))]
    n_classes = max(1, len(groups))

    grid = np.full((y_max, x_max), np.nan, float)
    for (y, x), codes in cells.items():
        vals, counts = np.unique(codes, return_counts=True)
        grid[y-1, x-1] = float(vals[np.argmax(counts)])

    cycle = plt.rcParams.get('axes.prop_cycle')
    cycle = cycle.by_key().get('color', []) if cycle else []
    if not cycle:
        cycle = plt.get_cmap('tab10').colors
    colors = [cycle[i % len(cycle)] for i in range(n_classes)]
    cmap = ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))

    x_edges = np.arange(0, x_max + 1)
    y_edges = np.arange(0, y_max + 1)
    ax.pcolormesh(x_edges, y_edges, grid, cmap=cmap, vmin=-0.5, vmax=n_classes-0.5, shading='flat')

    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max); ax.set_aspect('equal')
    ax.set_xticks(np.arange(0.5, x_max + 0.5, 1.0))
    ax.set_yticks(np.arange(0.5, y_max + 0.5, 1.0))
    ax.set_xticklabels([str(i) for i in range(1, x_max + 1)])
    ax.set_yticklabels([str(i) for i in range(1, y_max + 1)])
    ax.set_xticks(np.arange(0, x_max + 1), minor=True)
    ax.set_yticks(np.arange(0, y_max + 1), minor=True)
    ax.grid(True, which='minor'); ax.grid(False, which='major')
    ax.tick_params(which='minor', length=0)
    ax.set_xlabel(x_attr); ax.set_ylabel(y_attr)
    ax.tick_params(axis='both', which='major', labelsize=6)
    for xt in ax.get_xticklabels():
        xt.set_rotation(90); xt.set_horizontalalignment('center')

    present = np.unique(grid[~np.isnan(grid)]).astype(int) if np.any(~np.isnan(grid)) else []
    if len(groups) > 1 and len(present) > 0:
        handles = [Patch(label=class_names[c], facecolor=cmap(c), edgecolor='black', linewidth=0.8) for c in present]
        ax.legend(handles=handles, frameon=True, loc='lower left', mode='expand',
                  borderaxespad=0, bbox_to_anchor=(0, 1.02, 1, 0.2),
                  handlelength=0.7, handletextpad=0.45, ncol=len(present), fancybox=True)

    fig.tight_layout()
    return fig, ax

def plot_parity(result):
    plt.close('all')
    fig, ax = plt.subplots()

    y_true = getattr(result.meta, 'y_true', None)
    y_pred = getattr(result.meta, 'y_pred', None)
    if y_true is None or y_pred is None:
        ax.text(0.5, 0.5, "y_true or y_pred not found in result.meta",
                ha='center', va='center', transform=ax.transAxes)
        fig.tight_layout()
        return fig, ax

    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    m = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[m], yp[m]

    grades = np.asarray(yt, dtype=int)
    grades = np.clip(grades, 0, 5)

    pos = np.arange(6)
    data = [yp[grades == g] for g in pos]
    ax.boxplot(
        data,
        positions=pos,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="none", edgecolor="black", linewidth=1.4),
        medianprops=dict(linewidth=1.6),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(markersize=4, alpha=0.4),
        zorder=2,
        showfliers=False
    )

    ax.set_xlim(-0.5, 5.5)
    ax.set_xticks(pos)
    ax.set_ylim(0, 5)

    target = getattr(result.params, 'target', None)
    xlab = f"Actual {target}" if target is not None else "Actual"
    ylab = f"Predicted {target}" if target is not None else "Predicted"
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

    fig.tight_layout()
    return fig, ax
