import numpy as np
from signalearn.general_utility import *
import pickle
from scipy.fft import *
from scipy.interpolate import UnivariateSpline
import math

def load(filepath):

    with open(filepath, "rb") as file:
        return pickle.load(file)

def save(instance, filepath):

    with open(filepath, "wb") as file:
        pickle.dump(instance, file)

def get_labels(x):
    xlabel = getattr(x, "label", None) or "x"
    xunit = getattr(x, "unit", None) or ""

    return xlabel, xunit

def new_sample(sample, updates=None):
    params = {k: f.values for k, f in sample.fields.items()}
    labels = {k: f.label for k, f in sample.fields.items()}
    units = {k: f.unit for k, f in sample.fields.items()}
    aliases = {k: f.alias for k, f in sample.fields.items()}
    if updates:
        params.update(updates)
    params["labels"] = labels
    params["units"] = units
    params["aliases"] = aliases
    return sample.__class__(params)

def as_fields(y):
    if isinstance(y, (list, tuple)):
        return list(y)
    return [y]

def get_sample_rate(x):
    fields = x.fields if hasattr(x, "fields") else [x]
    for f in fields:
        vals = np.asarray(getattr(f, "values", f), dtype=float).ravel()
        if vals.size < 2:
            continue
        d = np.diff(vals)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            return float(1.0 / np.median(d))
    return float("nan")
