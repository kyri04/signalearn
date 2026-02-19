import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve, auc
from signalearn.utility import *

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    fig.savefig(f"{filename}.{extension}", bbox_inches='tight', dpi=dpi)

def plot_signal(x, y):
    plt.close('all')
    fig, ax = plt.subplots()

    xlabel,xunit = get_labels(x)
    y_list = list(y) if isinstance(y, (list, tuple)) else [y]
    ylabel,yunit = get_labels(y_list[0])

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"

    if len(y_list) > 1:
        for yi in y_list:
            lab = getattr(yi, "alias", None) or get_labels(yi)[0]
            ax.plot(x, yi, lw=1, label=lab)
        ax.legend()
    else:
        ax.plot(x, y_list[0], lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xunit:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if not yunit:
        ax.tick_params(axis="y", which="both", labelleft=False)

    fig.tight_layout()
    return fig, ax

def plot_mean(x, y):
    plt.close('all')
    fig, ax = plt.subplots()

    xlabel, xunit = get_labels(x)
    y_list = list(y) if isinstance(y, (list, tuple)) else [y]
    ylabel, yunit = get_labels(y_list[0])
    yunits = [get_labels(yi)[1] for yi in y_list]
    if not all(u == yunits[0] for u in yunits):
        yunit = ""

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"

    x_vals = np.asarray(x.fields[0].values if hasattr(x, "fields") else x.values, dtype=float).ravel()
    line_handles = []
    band_colors = []
    for yi in y_list:
        y_vals = np.asarray([np.asarray(f.values, dtype=float).ravel() for f in yi.fields], dtype=float)
        mean = np.mean(y_vals, axis=0)
        std = np.std(y_vals, axis=0)
        lo = mean - std
        hi = mean + std
        lab = getattr(yi, "alias", None) or get_labels(yi)[0]
        line = ax.plot(x_vals, mean, label=f"{lab} Mean")[0]
        line_handles.append(line)
        band_colors.append(line.get_color())
        ax.fill_between(x_vals, lo, hi, color=line.get_color(), alpha=0.15, label="_nolegend_")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xunit:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if not yunit:
        ax.tick_params(axis="y", which="both", labelleft=False)
    band_handles = tuple(Patch(facecolor=c, edgecolor="none", alpha=0.15) for c in band_colors)
    band_handle = band_handles[0] if len(band_handles) == 1 else band_handles
    handles = line_handles + [band_handle]
    labels = [h.get_label() for h in line_handles] + ["±1 Std"]
    handler_map = {tuple: HandlerTuple(ndivide=None)} if isinstance(band_handle, tuple) else None
    ax.legend(handles=handles, labels=labels, handler_map=handler_map)

    fig.tight_layout()
    return fig, ax

def plot_median(x, y):
    plt.close('all')
    fig, ax = plt.subplots()

    xlabel, xunit = get_labels(x)
    y_list = list(y) if isinstance(y, (list, tuple)) else [y]
    ylabel, yunit = get_labels(y_list[0])
    yunits = [get_labels(yi)[1] for yi in y_list]
    if not all(u == yunits[0] for u in yunits):
        yunit = ""

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"

    x_vals = np.asarray(x.fields[0].values if hasattr(x, "fields") else x.values, dtype=float).ravel()
    line_handles = []
    band_colors = []
    for yi in y_list:
        y_vals = np.asarray([np.asarray(f.values, dtype=float).ravel() for f in yi.fields], dtype=float)
        median = np.median(y_vals, axis=0)
        q25 = np.percentile(y_vals, 25, axis=0)
        q75 = np.percentile(y_vals, 75, axis=0)
        lab = getattr(yi, "alias", None) or get_labels(yi)[0]
        line = ax.plot(x_vals, median, label=f"{lab} Median")[0]
        line_handles.append(line)
        band_colors.append(line.get_color())
        ax.fill_between(x_vals, q25, q75, color=line.get_color(), alpha=0.15, label="_nolegend_")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xunit:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if not yunit:
        ax.tick_params(axis="y", which="both", labelleft=False)
    band_handles = tuple(Patch(facecolor=c, edgecolor="none", alpha=0.15) for c in band_colors)
    band_handle = band_handles[0] if len(band_handles) == 1 else band_handles
    handles = line_handles + [band_handle]
    labels = [h.get_label() for h in line_handles] + ["IQR"]
    handler_map = {tuple: HandlerTuple(ndivide=None)} if isinstance(band_handle, tuple) else None
    ax.legend(handles=handles, labels=labels, handler_map=handler_map)

    fig.tight_layout()
    return fig, ax

def plot_scatter(x, y, by=None):
    plt.close('all')
    fig, ax = plt.subplots()

    if hasattr(x, "_dataset"):
        from signalearn.learning_utility import get as align_get
        dataset = x._dataset
        if hasattr(y, "_dataset") and y._dataset is not dataset:
            y = align_get(dataset, y)
        if by is not None and hasattr(by, "_dataset") and by._dataset is not dataset:
            by = align_get(dataset, by)

    xlabel, xunit = get_labels(x)
    ylabel, yunit = get_labels(y)

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"

    xv = np.asarray([f.values for f in x.fields], dtype=float)
    yv = np.asarray([f.values for f in y.fields], dtype=float)
    if by is None:
        ax.scatter(xv, yv, s=20, alpha=0.7, edgecolors="0.2", linewidths=0.5)
    else:
        groups = np.asarray([f.values for f in by.fields], dtype=object)
        group_list = groups.tolist()

        def unique_order(vals):
            out = []
            for v in vals:
                if not any(v == u for u in out):
                    out.append(v)
            return out

        def sort_key(v):
            if v is None:
                return (2, "")
            try:
                f = float(v)
                if np.isnan(f):
                    return (2, "nan")
                return (0, f)
            except Exception:
                return (1, str(v))

        order = sorted(unique_order(group_list), key=sort_key)
        cycle = plt.rcParams.get("axes.prop_cycle", None)
        palette = cycle.by_key().get("color", []) if cycle is not None else []
        if not palette:
            palette = [f"C{i}" for i in range(max(1, len(order)))]
        colors = [palette[i % len(palette)] for i in range(len(order))]

        group_idx = []
        for gv in group_list:
            idx = 0
            for i, g in enumerate(order):
                if gv == g:
                    idx = i
                    break
            group_idx.append(idx)

        rng = np.random.default_rng()
        draw_order = rng.permutation(len(group_list))
        point_colors = [colors[group_idx[i]] for i in draw_order]
        ax.scatter(
            xv[draw_order],
            yv[draw_order],
            s=20,
            alpha=0.7,
            edgecolors="0.2",
            linewidths=0.5,
            c=point_colors,
        )

        for i, g in enumerate(order):
            ax.scatter([], [], s=20, alpha=0.7, edgecolors="0.2", linewidths=0.5, c=[colors[i]], label=str(g))
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if not xunit:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if not yunit:
        ax.tick_params(axis="y", which="both", labelleft=False)

    fig.tight_layout()
    return fig, ax

def plot_scatter3D(x, y, z, by=None):
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    if hasattr(x, "_dataset"):
        from signalearn.learning_utility import get as align_get
        dataset = x._dataset
        if hasattr(y, "_dataset") and y._dataset is not dataset:
            y = align_get(dataset, y)
        if hasattr(z, "_dataset") and z._dataset is not dataset:
            z = align_get(dataset, z)
        if by is not None and hasattr(by, "_dataset") and by._dataset is not dataset:
            by = align_get(dataset, by)

    xlabel, xunit = get_labels(x)
    ylabel, yunit = get_labels(y)
    zlabel, zunit = get_labels(z)

    if xunit:
        xlabel = f"{xlabel} ({xunit})"
    if yunit:
        ylabel = f"{ylabel} ({yunit})"
    if zunit:
        zlabel = f"{zlabel} ({zunit})"

    xv = np.asarray([f.values for f in x.fields], dtype=float)
    yv = np.asarray([f.values for f in y.fields], dtype=float)
    zv = np.asarray([f.values for f in z.fields], dtype=float)

    if by is None:
        ax.scatter(xv, yv, zv, s=20, alpha=0.7, edgecolors="0.2", linewidths=0.5)
    else:
        groups = np.asarray([f.values for f in by.fields], dtype=object)
        group_list = groups.tolist()

        def unique_order(vals):
            out = []
            for v in vals:
                if not any(v == u for u in out):
                    out.append(v)
            return out

        def sort_key(v):
            if v is None:
                return (2, "")
            try:
                f = float(v)
                if np.isnan(f):
                    return (2, "nan")
                return (0, f)
            except Exception:
                return (1, str(v))

        order = sorted(unique_order(group_list), key=sort_key)
        cycle = plt.rcParams.get("axes.prop_cycle", None)
        palette = cycle.by_key().get("color", []) if cycle is not None else []
        if not palette:
            palette = [f"C{i}" for i in range(max(1, len(order)))]
        colors = [palette[i % len(palette)] for i in range(len(order))]

        group_idx = []
        for gv in group_list:
            idx = 0
            for i, g in enumerate(order):
                if gv == g:
                    idx = i
                    break
            group_idx.append(idx)

        rng = np.random.default_rng()
        draw_order = rng.permutation(len(group_list))
        point_colors = [colors[group_idx[i]] for i in draw_order]
        ax.scatter(
            xv[draw_order],
            yv[draw_order],
            zv[draw_order],
            s=20,
            alpha=0.7,
            edgecolors="0.2",
            linewidths=0.5,
            c=point_colors,
        )

        for i, g in enumerate(order):
            ax.scatter([], [], [], s=20, alpha=0.7, edgecolors="0.2", linewidths=0.5, c=[colors[i]], label=str(g))
        ax.legend()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)

    if not xunit:
        ax.tick_params(axis="x", which="both", labelbottom=False)
    if not yunit:
        ax.tick_params(axis="y", which="both", labelleft=False)
    if not zunit:
        ax.tick_params(axis="z", which="both", labelleft=False)

    fig.tight_layout()
    return fig, ax

def plot_confusion_matrix(predictions):
    plt.close('all')
    fig, ax = plt.subplots()
    ax.grid(False)

    y_true = np.asarray([f.values for f in predictions.y_true.fields], dtype=object)
    y_pred = np.asarray([f.values for f in predictions.y_pred.fields], dtype=object)

    labels = np.unique(np.concatenate([y_true, y_pred], axis=0))
    idx = {lab: i for i, lab in enumerate(labels.tolist())}

    cm = np.zeros((labels.size, labels.size), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1

    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(labels.size))
    ax.set_yticks(np.arange(labels.size))
    ax.set_xticklabels([str(l) for l in labels], rotation=45, ha="right")
    ax.set_yticklabels([str(l) for l in labels])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    vmax = int(cm.max()) if cm.size else 0
    thresh = vmax / 2.0 if vmax else 0.0
    for i in range(labels.size):
        for j in range(labels.size):
            v = int(cm[i, j])
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > thresh else "black")

    fig.tight_layout()
    return fig, ax

def plot_predicted_vs_true(predictions):
    plt.close('all')
    fig, ax = plt.subplots()

    y_true = np.asarray([f.values for f in predictions.y_true.fields], dtype=float)
    y_pred = np.asarray([f.values for f in predictions.y_pred.fields], dtype=float)

    ax.scatter(y_true, y_pred, s=20, alpha=0.7, edgecolors="0.2", linewidths=0.5)

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))
    ax.plot([lo, hi], [lo, hi], color="0.4", lw=1)

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")

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

    ax.plot(best[2], best[3], color="green", label=f"Best AUC={best[1]:.3f}")
    ax.plot(worst[2], worst[3], color="red", label=f"Worst AUC={worst[1]:.3f}")
    ax.plot(grid, mean_tpr, color="black", label=f"Mean AUC={mean_auc:.3f}")

    ax.plot([0, 1], [0, 1], ls="--", lw=1, color="0.5")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    pad = 0.051
    ax.set_xlim(-pad, 1 + pad)
    ax.set_ylim(-pad, 1 + pad)
    ax.legend(loc="lower right")

    fig.tight_layout()
    return fig, ax
