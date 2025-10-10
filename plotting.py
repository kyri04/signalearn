import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from signalearn.learning_utility import reduce, scale
from signalearn.utility import *
from signalearn.preprocess import func_y
from sklearn.metrics import roc_curve, roc_auc_score

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
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
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

def plot_point(point, func=None):
    plt.close('all')
    fig, ax = plt.subplots()

    ax.set_yticklabels([])
    ax.set_xlabel(f'{point.xlabel} ({point.xunit})')
    ax.set_ylabel(point.ylabel if func is None else func.__name__ + ' ' + point.ylabel)
    ax.plot(point.x, point.y if func is None else func(point.y))

    fig.tight_layout()
    return fig, ax

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
    return fig, ax

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
    return fig, ax

def plot_distribution(points, func=np.mean):
    plt.close('all')
    fig, ax = plt.subplots()

    values = [func(point.y) for point in points if point.y is not None and len(point.y) > 0]

    ax.hist(values, bins=int(np.sqrt(len(values))))
    ax.set_xlabel(f'{func.__name__} {points[0].ylabel}')
    ax.set_ylabel('Frequency')

    fig.tight_layout()
    return fig, ax

def save_figure(filename, dpi=300, figsize=None, fig: Figure = None):
    if fig is None:
        fig = plt.gcf()
    if figsize is not None:
        fig.set_size_inches(figsize)
    os.makedirs('plots', exist_ok=True)
    save(fig, f"plots/{name_from_path(filename)}.pkl")
    fig.savefig(f"plots/{filename}", bbox_inches='tight', dpi=dpi)

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
