import numpy as np
import csv
from signalearn.general_utility import *
import pickle
from scipy.fft import *
from scipy.interpolate import UnivariateSpline
from types import SimpleNamespace
import math

def _label_unit(p, attr):
    units = getattr(p, "units", {}) or {}
    originals = getattr(p, "original_headers", {}) or {}
    u = units.get(attr, "") or ""
    h = originals.get(attr, attr)
    if u:
        h = re.sub(rf"\s*[\(\[]\s*{re.escape(u)}\s*[\)\]]\s*$", "", str(h))
    return h, u

def set_active(points, x, y):
    pts = points if isinstance(points, (list, tuple)) else [points]

    def block_for_axis(p, x_name, y_names):
        ax = np.asarray(getattr(p, x_name)); 
        if ax.ndim != 1: raise ValueError("x must be 1D")
        mats, labs, units = [], [], []
        for feat in y_names:
            arr = np.asarray(getattr(p, feat))
            if arr.ndim != 1 or arr.shape[0] != ax.shape[0]:
                raise ValueError(f"feature '{feat}' must be 1D and match x length")
            mats.append(arr)
            lab, uni = _label_unit(p, feat); labs.append(lab); units.append(uni)
        y_block = np.vstack(mats)  # (C,N)
        xl, xu = _label_unit(p, x_name)
        return ax, y_block, xl, xu, labs, units

    for p in pts:
        if isinstance(x, str):
            y_names = [y] if isinstance(y, str) else list(y)
            ax, yblk, xl, xu, ylabs, yunits = block_for_axis(p, x, y_names)
            p.x      = [ax]
            p.y      = [yblk]          # <-- list containing the 2D block
            p.xlabel = [xl]
            p.xunit  = [xu]
            p.ylabel = [ylabs]         # <-- list of lists
            p.yunit  = [yunits]        # <-- list of lists
        else:
            if not isinstance(x, (list, tuple)) or not isinstance(y, (list, tuple)):
                raise ValueError("x must be str or list; y must match (str/list or list-of-lists)")
            if any(not isinstance(ys, (list, tuple)) for ys in y) or len(x) != len(y):
                raise ValueError("when x is a list, y must be a list of lists with equal length")
            xs, ys, xls, xus, yls, yus = [], [], [], [], [], []
            for xn, ynames in zip(x, y):
                ax, yblk, xl, xu, ylabs, yunits = block_for_axis(p, xn, list(ynames))
                xs.append(ax); ys.append(yblk)
                xls.append(xl); xus.append(xu)
                yls.append(ylabs); yus.append(yunits)
            p.x      = xs
            p.y      = ys              # list of 2D blocks
            p.xlabel = xls
            p.xunit  = xus
            p.ylabel = yls             # list of lists
            p.yunit  = yus             # list of lists

def read_map(map_path):
    with open(map_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        map = [row for row in reader]
    return map

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

def find_unique(points, attr):

    if isinstance(attr, str):
        unique_vals = set([getattr(point, attr) for point in points])
    elif isinstance(attr, list):
        unique_vals = set(["_".join(str(getattr(point, a)) for a in attr) for point in points])
    else:
        raise ValueError("label must be either a string or a list")

    return list(unique_vals)

def count_unique(points, attr):
    return len(find_unique(points, attr))

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
