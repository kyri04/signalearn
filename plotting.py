import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from signalearn.utility import *

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    fig.savefig(f"{filename}.{extension}", bbox_inches='tight', dpi=dpi)

def plot_signal(x, y):
    plt.close('all')
    fig, ax = plt.subplots()

    xlabel,xunit = get_labels(x)
    ylabel,yunit = get_labels(y)

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"

    ax.plot(x, y, lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    fig.tight_layout()
    return fig, ax

def plot_roc(predictions, by=None):
    plt.close('all')
    fig, ax = plt.subplots()

    y_true = np.asarray([f.values for f in predictions.y_true.fields], dtype=object)
    y_score = np.asarray([f.values for f in predictions.y_score.fields], dtype=float)

    labels = np.unique(y_true)
    if labels.size != 2:
        raise ValueError("ROC curve requires binary targets.")
    pos_label = labels[1]

    if by is None:
        groups = [(None, np.arange(y_true.shape[0], dtype=int))]
    else:
        keys = np.asarray([f.values for f in by.fields], dtype=object)
        uniq = np.unique(keys)
        groups = [(k, np.flatnonzero(keys == k)) for k in uniq]

    grid = np.linspace(0.0, 1.0, 300)
    curves = []
    for k, idx in groups:
        fpr, tpr, _ = roc_curve(y_true[idx], y_score[idx], pos_label=pos_label)
        a = float(auc(fpr, tpr))
        tpr_i = np.interp(grid, fpr, tpr)
        tpr_i[0] = 0.0
        tpr_i[-1] = 1.0
        curves.append((k, a, fpr, tpr, tpr_i))

    best = max(curves, key=lambda c: c[1])
    worst = min(curves, key=lambda c: c[1])
    mean_tpr = np.mean([c[4] for c in curves], axis=0)
    mean_auc = float(auc(grid, mean_tpr))

    for k, a, fpr, tpr, _ in curves:
        if (k, a) == (best[0], best[1]) or (k, a) == (worst[0], worst[1]):
            continue
        ax.plot(fpr, tpr, lw=1.0, alpha=0.25, color="0.75")

    ax.plot(best[2], best[3], lw=2.0, color="green", label=f"Best AUC={best[1]:.3f}")
    ax.plot(worst[2], worst[3], lw=2.0, color="red", label=f"Worst AUC={worst[1]:.3f}")
    ax.plot(grid, mean_tpr, lw=2.0, color="black", label=f"Mean AUC={mean_auc:.3f}")

    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.5")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax
