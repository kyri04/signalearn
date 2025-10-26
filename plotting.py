import os
import matplotlib.pyplot as plt
import numpy as np
from signalearn.learning_utility import reduce, scale
from signalearn.utility import *
from signalearn.preprocess import func_y
from sklearn.metrics import roc_curve, roc_auc_score
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    os.makedirs('plots/export', exist_ok=True)
    fig.savefig(f"plots/{filename}.{extension}", bbox_inches='tight', dpi=dpi)

def plot_importances(result, points=None, func=None):
    plt.close('all')
    fig, ax = plt.subplots()
    if isinstance(result, list):
        importances = []
        first = None
        for r in result:
            fi = getattr(r.results, "feature_importances", None)
            if fi is not None:
                importances.append(np.asarray(fi, dtype=float))
                if first is None:
                    first = r

        imp = np.mean(importances, axis=0)
        x_imp = np.asarray(first.params.mean_point.x, dtype=float)
        xlabel = f"{first.params.mean_point.xlabel} ({first.params.mean_point.xunit})"
    else:
        fi = getattr(result.results, "feature_importances", None)
        imp = np.asarray(fi, dtype=float)
        x_imp = np.asarray(points[0].x, dtype=float)
        xlabel = f"{points[0].xlabel} ({points[0].xunit})"

    ax.plot(x_imp, imp, label="Feature importances")

    if points:

        Ys = []
        for p in points:
            Ys.append(func(p.y))

        Y = np.vstack(Ys)
        med = np.median(Y, axis=0)

        imp_min, imp_max = float(np.min(imp)), float(np.max(imp))
        med_min, med_max = float(np.min(med)), float(np.max(med))
        if med_max > med_min:
            scale = (imp_max - imp_min) / (med_max - med_min) if imp_max > imp_min else 1.0
            offset = imp_min - scale * med_min
            med_scaled = scale * med + offset
        else:
            med_scaled = np.full_like(med, imp_min)

        lbl = f"Median {points[0].ylabel}" if func is None else f"Median {func.__name__}({points[0].ylabel})"
        ax.plot(x_imp, med_scaled, linestyle=":", color="0", label=lbl)

    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Importance")
    if points:
        ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_pca(points, label=None, n_components=2):
    plt.close('all')
    y = [point.y for point in points]

    labels = None
    savename = ""
    if isinstance(label, str):
        labels = [getattr(point, label) for point in points] if label else None
        savename = label
    if isinstance(label, list):
        labels = ["_".join(str(getattr(point, attr)) for attr in label) for point in points]
        savename = ""
        for l in label:
            savename += f"-{l}"

    if labels:
        unique_labels = list(set(labels))
        label_to_color = {label: idx for idx, label in enumerate(unique_labels)}
        colors = [label_to_color[label] for label in labels]
        cmap = plt.colormaps.get_cmap('viridis').resampled(len(unique_labels))
    else:
        colors = None
        cmap = None

    y_reduced = reduce(y, n_components)

    if n_components == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(
            y_reduced[:, 0],
            y_reduced[:, 1],
            alpha=0.7,
            edgecolors='k',
            c=colors,
            cmap=cmap
        )
        ax.set_title("Principal Component Analysis (2D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True)

    else:  # n_components == 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(
            y_reduced[:, 0],
            y_reduced[:, 1],
            y_reduced[:, 2],
            alpha=0.7,
            edgecolors='k',
            c=colors,
            cmap=cmap
        )
        ax.set_title("Principal Component Analysis (3D)")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_zlabel("Principal Component 3")

    if label is not None and labels:
        legend1 = ax.legend(
            handles=[plt.Line2D([0], [0], marker='o', color='w', label=unique_label,
                                markersize=10, markerfacecolor=cmap(idx / len(unique_labels)))
                     for idx, unique_label in enumerate(unique_labels)],
            title="Labels"
        )
        ax.add_artist(legend1)

    fig.tight_layout()
    return fig, ax

def _norm(point):
    xs = point.x
    ys = point.y
    xl = point.xlabel
    xu = point.xunit
    yl = point.ylabel
    yu = point.yunit

    if isinstance(xs, np.ndarray): xs = [xs]
    if isinstance(ys, np.ndarray): ys = [ys if ys.ndim == 2 else ys[None, :]]
    if isinstance(xl, str): xl = [xl]
    if isinstance(xu, str): xu = [xu]
    if isinstance(yl, list) and yl and isinstance(yl[0], str): yl = [yl]
    if isinstance(yu, list) and yu and isinstance(yu[0], str): yu = [yu]
    return xs, ys, xl, xu, yl, yu

def plot_point(point, func=None):
    plt.close('all')
    xs, ys, xlabs, xunits, ylabs, yunits = _norm(point)
    n = len(xs)

    fig, axes = plt.subplots(n, 1, figsize=(6, 3*n))
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        X = np.asarray(xs[i])
        Y = ys[i]
        if Y.ndim == 1: Y = Y[None, :]
        rows = [func(row) if func is not None else row for row in Y]
        for r in rows:
            ax.plot(X, r)
        ax.set_yticks([])
        xlab = xlabs[i] if xlabs[i] is not None else ""
        xunit = xunits[i] if xunits[i] is not None else ""
        ax.set_xlabel(f"{xlab} ({xunit})" if xunit else xlab)

        ylab_list = ylabs[i] if ylabs[i] is not None else []
        yunit_list = yunits[i] if yunits[i] is not None else []
        if yunit_list and len(yunit_list) == len(ylab_list):
            ylab = ", ".join(f"{a}" for a in ylab_list)
        else:
            ylab = ", ".join(ylab_list)
        if ylab:
            prefix = (func.__name__ + " ") if func is not None else ""
            ax.set_ylabel(prefix + ylab)

        if len(ylab_list) > 1:
            ax.legend(ylab_list, loc="best", frameon=False)

    fig.tight_layout()
    return fig, axes

def plot_points_range(points, func=None, q=(0.25, 0.5, 0.75)):
    plt.close('all')
    fig, ax = plt.subplots()

    X = np.asarray(points[0].x)
    Y = np.vstack([(p.y if func is None else func(p.y)) for p in points])

    Y[~np.isfinite(Y)] = np.nan
    q1, med, q3 = np.nanquantile(Y, q, axis=0)

    m = np.isfinite(q1) & np.isfinite(med) & np.isfinite(q3)
    ax.fill_between(X[m], q1[m], q3[m], alpha=0.25, label='IQR')
    ax.plot(X[m], med[m], lw=1.5, label='Median')

    ax.set_yticks([])
    ax.set_xlabel(f'{points[0].xlabel} ({points[0].xunit})')
    ylab = points[0].ylabel if func is None else f'{func.__name__} {points[0].ylabel}'
    ax.set_ylabel(ylab)
    ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_scaled(points, idx=0):
    ys_scaled = scale(np.array([point.y for point in points]))

    plt.close('all')
    fig, ax = plt.subplots()
    ax.set_yticklabels([])
    ax.set_xlabel(f'{points[idx].xlabel} ({points[idx].xunit})')
    ax.set_ylabel(f'Standardised {points[idx].ylabel}')
    ax.plot(points[idx].x, ys_scaled[0])

    fig.tight_layout()
    return plt

def plot_points(points, func=None, offset=0.1):
    plt.close('all')
    fig, ax = plt.subplots()
    for i, p in enumerate(points):
        y = p.y if func is None else func(p.y)
        if offset:
            y = y + i * offset
        ax.plot(p.x, y, alpha=0.7)

    ax.set_yticks([])
    ylab = p.ylabel if func is None else f"{func.__name__} {p.ylabel}"
    ax.set_ylabel(ylab)
    ax.set_xlabel(f'{points[0].xlabel} ({points[0].xunit})')
    ax.margins(x=0)

    fig.tight_layout()
    return fig, ax

def plot_func(points, func=np.mean):
    plt.close('all')
    attr, first_val = find_same_attribute(points)
    y = func_y(points, func=func)

    fig, ax = plt.subplots()
    ax.set_xlabel(f'{points[0].xlabel} ({points[0].xunit})')
    ax.set_ylabel(f'{points[0].ylabel}')
    ax.plot(points[0].x, y)

    # ax.set_title(f"{func.__name__.capitalize()} of Points with {attr}={first_val}")

    fig.tight_layout()
    return fig, ax

def plot_func_difference(points_a, points_b, func=np.mean):
    plt.close('all')
    fig, ax = plt.subplots()
    y_a = func_y(points_a, func=func)
    y_b = func_y(points_b, func=func)
    y = y_a - y_b

    ax.set_xlabel(f'{points_a[0].xlabel} ({points_a[0].xunit})')
    ax.set_ylabel(f'{points_a[0].ylabel}')
    ax.plot(points_a[0].x, y)

    # ax.set_title(f"{func.__name__.capitalize()} Difference")

    fig.tight_layout()
    return plt

def plot_distribution(points, yfunc=np.mean, func=None):
    plt.close('all')
    fig, ax = plt.subplots()
    is_grouped = isinstance(points, (list, tuple)) and points and isinstance(points[0], (list, tuple))
    groups = points if is_grouped else [points]

    def values_from(group):
        vals = [yfunc(p.y) for p in group if getattr(p, "y", None) is not None and len(p.y) > 0]
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
    y_label = getattr(first_point, "ylabel", "Intensity")
    func_name = func.__name__ if callable(func) else ""
    x_label_core = f"{yfunc.__name__} {y_label}".strip()

    ax.set_xlabel(f"{func_name}({x_label_core})" if func_name else x_label_core)
    ax.set_ylabel("Frequency")
    if len(groups) > 1:
        ax.legend()
    fig.tight_layout()

    return fig, ax

def plot_rocs(results):
    plt.close('all')
    fig, ax = plt.subplots()
    valid = [r.group_results for r in results
             if (r.group_results.y_true is not None and r.group_results.y_score is not None)]

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

def plot_grid(points, xattr, yattr, sizeattr):
    plt.close('all')
    fig, ax = plt.subplots()

    is_grouped = isinstance(points, (list, tuple)) and points and isinstance(points[0], (list, tuple))
    groups = points if is_grouped else [points]

    cells = {}

    grid_size = getattr(groups[0][0], sizeattr)
    x_max = grid_size[0]
    y_max = grid_size[1]

    for gi, group in enumerate(groups):
        for p in group:
            x = int(getattr(p, xattr)); y = int(getattr(p, yattr))
            # x_max = max(x_max, x); y_max = max(y_max, y)
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
    ax.set_xlabel(xattr); ax.set_ylabel(yattr)
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


