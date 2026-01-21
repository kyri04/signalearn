import matplotlib.pyplot as plt
import numpy as np
from signalearn.utility import *
from sklearn.metrics import roc_curve, roc_auc_score

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    fig.savefig(f"{filename}.{extension}", bbox_inches='tight', dpi=dpi)

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