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

def get_axes_labels(point, *attrs):
    labels = []
    for attr in attrs:
        unit = point.units.get(attr, "")
        lbl = point.labels.get(attr, attr)
        if unit:
            lbl = f"{lbl} ({unit})"
        labels.append(lbl)
    return labels

def get_sample_rate(points, x_attr):
    def _rate(point):
        x = getattr(point, x_attr, None)
        if x is None:
            return None

        try:
            arr = np.asarray(x, dtype=float)
        except Exception:
            return None

        if arr.ndim == 0 or arr.size < 2:
            return None

        diffs = np.diff(arr)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size == 0:
            return None

        return float(1.0 / np.median(diffs))

    if isinstance(points, (list, tuple, set)):
        return [_rate(p) for p in points]
    return _rate(points)

def find_same_attribute(points):

    common_attrs = vars(points[0]).keys()

    for attr in common_attrs:
        if attr in ['xlabel', 'ylabel', 'xunit', 'yunit', ]: continue
        first_val = getattr(points[0], attr)
        
        try:
            if all(getattr(point, attr) == first_val for point in points):
                return attr, first_val
        except ValueError as e:
            continue

    return None, None

def calculate_filtered(points, filtered):
    removed_count = len(points) - len(filtered)
    removed_percentage = (removed_count / len(points)) * 100 if points else 0
    print(f"Removed {removed_percentage:.2f}% of points")

def update_points(points, point_class, params=None):

    new_points = []

    for point in points:

        new_params = point.__dict__
        if params is not None: new_params.update(params)

        new_points.append(point_class(new_params))
        
    return new_points

def update_directory(dir, point_class, params=None):

    for pfile in os.scandir(dir):
        points = load(pfile.path)
        update_points(points, point_class=point_class, params=params)
        save(points, pfile.path)

def combine_vals(points, attr, vals, new_val):

    for point in points:

        if any(v.lower() in getattr(point, attr).lower() for v in vals): setattr(point, attr, new_val)
        
    return points

def find_unique(points, attr):

    if isinstance(attr, str):
        unique_vals = set([getattr(point, attr) for point in points])
    elif isinstance(attr, list):
        unique_vals = set(["_".join(str(getattr(point, a)) for a in attr) for point in points])
    else:
        raise ValueError("label must be either a string or a list")

    return list(unique_vals)

def count_unique(points, attr):
    vals = set()
    for p in points:
        v = getattr(p, attr, None)

        if isinstance(v, np.ndarray):
            if v.shape == ():
                v = v.item()
            else:
                continue

        if isinstance(v, (list, tuple, dict)):
            continue

        if v is None:
            continue
        if isinstance(v, float) and np.isnan(v):
            continue

        vals.add(v)
    return len(vals)

def fit_spline(x, y, smooth=None):

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    order = np.argsort(x)
    x, y = x[order], y[order]

    if smooth is None:
        smooth = max(1e-8, 0.5 * len(x))
    f = UnivariateSpline(x, y, s=smooth)
    return f

def grid_num_to_xy(grid_number, total_cells, snake=False, zero_based=True):
    grid_number = int(grid_number)
    total_cells = int(total_cells)

    if zero_based:
        idx = grid_number
    else:
        idx = grid_number - 1

    n_cols = math.ceil(math.sqrt(total_cells))
    row = idx // n_cols
    col = idx % n_cols

    if snake and (row % 2 == 1):
        grid_x = n_cols - col
    else:
        grid_x = col + 1

    grid_y = row + 1
    return grid_x, grid_y

def group_points(points, group_attr=None):
    if group_attr is None:
        return [points]
    group_map = {}
    for p in points:
        g = getattr(p, group_attr, None)
        group_map.setdefault(g, []).append(p)
    return list(group_map.values())

def filter_numeric(points, attr):
    ok = []
    for i, p in enumerate(points):
        y = np.asarray(getattr(p, attr))
        if y.dtype.kind in "fiu":
            ok.append(p)
        else:
            print(f"Removing {p.name} with non-numeric {attr}")
    return ok

def set_each(points, new_attr, values):
    for point, value in zip(points, values):
        setattr(point, new_attr, value)
    return points

def set_meta(points, attr, unit=None, label=None):
    if not points:
        return points
    first = points[0]
    units = getattr(first, "units", None)
    labels = getattr(first, "labels", None)
    if unit is None and isinstance(units, dict):
        unit = units.get(attr, None)
    if label is None:
        label = pretty(attr)
    for p in points:
        u = getattr(p, "units", None)
        l = getattr(p, "labels", None)
        if isinstance(u, dict) and unit is not None:
            u[attr] = unit
        if isinstance(l, dict) and label is not None:
            l[attr] = label
    return points

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

def as_fields(y_attr):
    if isinstance(y_attr, (list, tuple)):
        return list(y_attr)
    return [y_attr]