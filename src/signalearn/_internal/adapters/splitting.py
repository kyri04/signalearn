from copy import deepcopy

import numpy as np


def native_splits(splitter, length, target, group):
    if not callable(getattr(splitter, "split", None)):
        raise TypeError("splitter must provide split(X, y=None, groups=None)")
    splitter = deepcopy(splitter)
    arguments = {}
    if target is not None:
        arguments["y"] = target
    if group is not None:
        arguments["groups"] = group
    return tuple(splitter.split(np.empty((length, 0)), **arguments))
