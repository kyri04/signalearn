import numpy as np
from signalearn.general_utility import *
import pickle
from scipy.fft import *
from scipy.interpolate import UnivariateSpline
from types import SimpleNamespace
from collections import defaultdict
import math

def filter(points, attr, val):
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

    return has

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

import numpy as np

def grid_boundaries(points, xattr, yattr, sizeattr, include_empty=True, exclude_edges=True):
    """
    1-based coords: x,y in [1..W],[1..H] where [W,H] = getattr(point, sizeattr).
    Returns N lists (one per group) of boundary points. 
    """
    # normalize to list-of-groups
    is_grouped = isinstance(points, (list, tuple)) and points and isinstance(points[0], (list, tuple))
    groups = points if is_grouped else [points]

    # grid size (W,H) from first point
    W, H = map(int, getattr(groups[0][0], sizeattr))

    # votes per 1-based cell
    cell_codes = {}
    for gi, group in enumerate(groups):
        for p in group:
            x = int(getattr(p, xattr)); y = int(getattr(p, yattr))
            if 1 <= x <= W and 1 <= y <= H:
                cell_codes.setdefault((y, x), []).append(gi)

    # build 0-based grid
    grid = np.full((H, W), np.nan)
    for (y, x), codes in cell_codes.items():
        y0, x0 = y-1, x-1
        vals, counts = np.unique(codes, return_counts=True)
        grid[y0, x0] = float(vals[np.argmax(counts)])

    # boundary mask (4-neighborhood)
    boundary = np.zeros_like(grid, dtype=bool)
    nbrs = [(-1,0),(1,0),(0,-1),(0,1)]
    for y0 in range(H):
        for x0 in range(W):
            cur = grid[y0, x0]
            if np.isnan(cur): 
                continue
            for dy, dx in nbrs:
                ny, nx = y0+dy, x0+dx
                if 0 <= ny < H and 0 <= nx < W:
                    nb = grid[ny, nx]
                    if (not np.isnan(nb) and nb != cur) or (include_empty and np.isnan(nb)):
                        boundary[y0, x0] = True; break
                else:
                    if include_empty:
                        boundary[y0, x0] = True; break

    # collect boundary points, excluding 1-based outer frame if requested
    out = []
    for gi, group in enumerate(groups):
        pts = []
        for p in group:
            x = int(getattr(p, xattr)); y = int(getattr(p, yattr))
            if not (1 <= x <= W and 1 <= y <= H):
                continue
            if exclude_edges and (x == 1 or x == W or y == 1 or y == H):
                continue
            if boundary[y-1, x-1]:
                pts.append(p)
        out.append(pts)
    return out
