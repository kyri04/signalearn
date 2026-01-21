import matplotlib.pyplot as plt
import numpy as np
from signalearn.utility import *
from sklearn.metrics import roc_curve, roc_auc_score

def save_plot(plot, filename, dpi=300, extension='pdf'):
    fig = plot[0]
    fig.savefig(f"{filename}.{extension}", bbox_inches='tight', dpi=dpi)

def plot_point(x, y):
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
