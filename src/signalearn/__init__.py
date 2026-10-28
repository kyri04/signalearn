from importlib.metadata import version as _version

from .data import Dataset, Field, Sample, SampleField
from .io import load_data, load_model, save_data, save_model
from .evaluation import accuracy, crossval, f1, mae, precision, r2, recall, rmse, sensitivity, specificity
from .learning import fit, predict
from .plotting import plot_box, plot_confusion_matrix, plot_hist, plot_image, plot_line, plot_scatter, save_plot, set_plot_style
from .preprocess import Folds, between, concat, crop, filter, folds, sample, slice, sort, split, take, transform, where, window

__version__ = _version("signalearn")

__all__ = ["__version__", "Dataset", "Field", "Sample", "SampleField", "load_data", "save_data",
           "save_model", "load_model", "save_plot",
           "Folds", "folds", "crossval", "split", "fit", "transform", "predict", "accuracy", "precision", "recall", "sensitivity",
           "specificity", "f1", "mae", "rmse", "r2", "plot_scatter", "plot_hist", "plot_box",
           "plot_image", "plot_line", "plot_confusion_matrix", "set_plot_style", "filter", "between", "where", "sort",
           "take", "sample", "concat", "slice", "crop", "window"]
