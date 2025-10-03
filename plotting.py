try:
    from cuml.internals.sklearn import patch_sklearn
    patch_sklearn()
except Exception:
    # cuml not installed or patch failed; proceed on CPU
    pass

import os
import matplotlib.pyplot as plt
import numpy as np
from signalearn.learning_utility import reduce, scale
from signalearn.utility import *
from sklearn.metrics import roc_curve, roc_auc_score

def plot_importances(result):
    plt.close('all')
    plt.figure()

    best_index = result.scores.index(max(result.scores))

    plt.plot(result.x_range, result.feature_importances[best_index])

    plt.xlabel(f'{result.points[0].xlabel} ({result.points[0].xunit})')
    plt.ylabel('Importance')
    # plt.title('Feature Importances')

    plt.tight_layout()
    plt.show()

def plot_pca(points, label=None, n_components=2):
    plt.close('all')
    if n_components not in [2, 3]:
        raise ValueError("n_components must be either 2 or 3.")
    y = [point.y for point in points]

    labels = None
    savename = ""
    if(isinstance(label, str)):
        labels = [getattr(point, label) for point in points] if label else None
        savename = label
    if(isinstance(label, list)):
        labels = ["_".join(str(getattr(point, attr)) for attr in label) for point in points]
        savename = ""
        for l in label: savename += f"-{l}"

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

    elif n_components == 3:
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

    plt.tight_layout()
    plt.show()

def plot_point(point, func=None):
    plt.close('all')

    plt.gca().set_yticklabels([])

    plt.xlabel(f'{point.xlabel} ({point.xunit})')
    plt.ylabel(point.ylabel if func is None else func.__name__ + ' ' + point.ylabel)
    plt.plot(point.x, point.y if func is None else func(point.y))

    plt.tight_layout()
    plt.show()

def plot_scaled(points, idx=0):

    ys_scaled = scale(np.array([point.y for point in points]))

    plt.close('all')

    plt.gca().set_yticklabels([])

    plt.xlabel(f'{points[idx].xlabel} ({points[idx].xunit})')
    plt.ylabel(f'Standardised {points[idx].ylabel}')
    plt.plot(points[idx].x, ys_scaled[0])

    plt.tight_layout()
    plt.show()

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
    plt.show()

def plot_func(points, func=np.mean):
    plt.close('all')
    attr, first_val = find_same_attribute(points)
    y = func_y(points, func=func)

    plt.figure()

    plt.xlabel(f'{points[0].xlabel} ({points[0].xunit})')
    plt.ylabel(f'{points[0].ylabel}')
    plt.plot(points[0].x, y)

    # plt.title(f"{func.__name__.capitalize()} of Points with {attr}={first_val}")

    plt.tight_layout()
    plt.show()

def plot_func_difference(points_a, points_b, func=np.mean):
    plt.close('all')
    y_a = func_y(points_a, func=func)
    y_b = func_y(points_b, func=func)

    y = y_a-y_b

    plt.figure()

    plt.xlabel(f'{points_a[0].xlabel} ({points_a[0].xunit})')
    plt.ylabel(f'{points_a[0].ylabel}')
    plt.plot(points_a[0].x, y)

    # plt.title(f"{func.__name__.capitalize()} Difference")

    plt.tight_layout()
    plt.show()

def plot_distribution(points, func=np.mean):
    plt.close('all')
    # Calculate the mean of y values for each point
    values = [func(point.y) for point in points if point.y is not None and len(point.y) > 0]

    # Plot the distribution of mean values
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=int(np.sqrt(len(values))))
    plt.xlabel(f'{func.__name__} {points[0].ylabel}')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def save_figure(filename, dpi=300, figsize=None):
    fig = plt.gcf()
    if(figsize != None): fig.set_size_inches(figsize)
    os.makedirs('plots', exist_ok=True)
    fig.savefig(f"plots/{filename}", bbox_inches='tight', dpi=dpi)

def plot_rocs(results):
    plt.close('all')
    valid = [r.group_results for r in results if (r.group_results.y_true is not None and r.group_results.y_score is not None)]

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

    plt.figure()
    for auc, fpr, tpr in roc_data:
        plt.plot(fpr, tpr, linewidth=0.8, alpha=0.2, color=(0.7, 0.7, 0.7))

    plt.plot([0, 1], [0, 1], linestyle=':', label=f"Baseline AUC = 0.500", linewidth=1.0, color=(0.5, 0.5, 0.5))

    plt.plot(best_fpr,  best_tpr,  linewidth=1.6, label=f"Best AUC = {best_auc:.3f}",  color='green')
    plt.plot(worst_fpr, worst_tpr, linewidth=1.6, label=f"Worst AUC = {worst_auc:.3f}", color='red')

    plt.plot(mean_grid, mean_tpr, linewidth=1.8, label=f"Mean AUC = {mean_auc:.3f}", color='black')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

def plot_learning_curve(xs, scores, xlabel="Training Data Fraction", ylabel="Score"):
    plt.close('all')
    plt.figure()

    plt.scatter(xs, scores)
    plt.plot(xs, fit_spline(xs, scores)(xs), linestyle="--", color="black")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.tight_layout()
    plt.show()