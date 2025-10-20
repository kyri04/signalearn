import numpy as np
import csv
from signalearn.general_utility import *
import pickle
from scipy.fft import *
from scipy.interpolate import UnivariateSpline
from types import SimpleNamespace
import math

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

    return unique_vals

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
