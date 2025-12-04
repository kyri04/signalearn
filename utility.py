import numpy as np
from signalearn.general_utility import *
import pickle
from scipy.fft import *
from scipy.interpolate import UnivariateSpline
from types import SimpleNamespace
from collections import defaultdict
import math

def filter(points, attr, val, includes=True):
    if isinstance(val, str):
        vals = [val.lower()]
    else:
        vals = [v.lower() for v in val]

    has, hasnt = [], []
    for point in points:
        attr_val = str(getattr(point, attr, "")).lower()
        if any(v in attr_val for v in vals):
            has.append(point)
        else:
            hasnt.append(point)

    if(includes):
        return has
    else:
        return hasnt

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

def make_namespace(d):
    return SimpleNamespace(**d)

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

def combine(points, match_id):
    groups = defaultdict(list)
    for p in points:
        if hasattr(p, match_id):
            k = getattr(p, match_id)
            if k is not None:
                groups[str(k)].append(p)

    out = []
    for k, grp in groups.items():
        if len(grp) < 2:
            continue

        params = {match_id: k}
        names = [getattr(p, "name", None) for p in grp if getattr(p, "name", None)]
        params["name"] = "+".join(sorted(set(names))) if names else str(k)

        meta = {"units", "labels", "name", match_id}
        attrs = set()
        for p in grp:
            attrs.update(a for a in getattr(p, "__dict__", {}) if a not in meta)

        units_acc, labels_acc = {}, {}

        for a in sorted(attrs):
            vals = [getattr(p, a) for p in grp if hasattr(p, a)]

            arrays, scalars = [], []
            for v in vals:
                if isinstance(v, (str, bytes)) or np.isscalar(v):
                    scalars.append(v)
                else:
                    arrays.append(np.asarray(v).ravel())

            if arrays:
                params[a] = np.concatenate(arrays, axis=0)
            else:
                if scalars and all(s == scalars[0] for s in scalars):
                    params[a] = scalars[0]
                else:
                    seen = set()
                    uniq = []
                    for s in scalars:
                        if s not in seen:
                            seen.add(s)
                            uniq.append(s)
                    params[a] = uniq

            for p in grp:
                u = getattr(p, "units", None)
                if isinstance(u, dict) and a in u and a not in units_acc:
                    units_acc[a] = u[a]
                l = getattr(p, "labels", None)
                if isinstance(l, dict) and a in l and a not in labels_acc:
                    labels_acc[a] = l[a]

        params["units"] = units_acc
        params["labels"] = labels_acc
        out.append(grp[0].__class__(params))

    return out

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

def set_all(points, new_attr, value):
    for point in points:
        setattr(point, new_attr, value)
    return points

def set_each(points, new_attr, values):
    for point, value in zip(points, values):
        setattr(point, new_attr, value)
    return points

def set_func(points, attr, func):
    name = getattr(func, "__name__", "")
    new_attr = f"{attr}_{name}" if name else attr
    for p in points:
        setattr(p, new_attr, func(getattr(p, attr)))
    set_meta(points, new_attr, unit=None, label=None)
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

def rename(points, old_attr, new_attr):
    if isinstance(points, (list, tuple)):
        pts = list(points)
    else:
        pts = [points]
    for p in pts:
        if hasattr(p, old_attr):
            setattr(p, new_attr, getattr(p, old_attr))
            delattr(p, old_attr)
        u = getattr(p, "units", None)
        if isinstance(u, dict) and old_attr in u:
            u[new_attr] = u.pop(old_attr)
        l = getattr(p, "labels", None)
        if isinstance(l, dict) and old_attr in l:
            l[new_attr] = l.pop(old_attr)
    return points

def entropy(y, eps=1e-9):
    y = np.asarray(y, float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    s = y.sum()
    if s <= 0:
        return 0.0
    p = y / s
    return float(-(p * np.log(p + eps)).sum())

def max_fraction(y, eps=1e-9):
    y = np.asarray(y, float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    total = y.sum()
    if total <= 0:
        return 0.0
    return float(y.max() / (total + eps))

def skewness(y, eps=1e-9):
    y = np.asarray(y, float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return 0.0
    m = y.mean()
    s = y.std()
    if s <= 0:
        return 0.0
    z = (y - m) / (s + eps)
    return float((z**3).mean())

def gini(y, eps=1e-9):
    y = np.asarray(y, float).ravel()
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    if y.size == 0:
        return 0.0
    y = np.sort(y)
    total = y.sum()
    if total <= 0:
        return 0.0
    n = y.size
    i = np.arange(1, n + 1, dtype=float)
    return float((2.0 * (i * y).sum() / (n * total)) - (n + 1.0) / n)
