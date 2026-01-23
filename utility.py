import numpy as np
from signalearn.general_utility import *
from signalearn.classes import Sample
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
    if updates:
        params.update(updates)
    params["labels"] = labels
    params["units"] = units
    return Sample(params)

def as_fields(y):
    if isinstance(y, (list, tuple)):
        return list(y)
    return [y]