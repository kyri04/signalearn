import re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from signalearn.classes import Series
from signalearn.general_utility import snake

_header_re = re.compile(r"^\s*(?P<base>[^\[(]+?)\s*(?:\((?P<u1>[^()]*)\)|\[(?P<u2>[^\[\]]*)\])?\s*$")

def parse_header(h):
    m = _header_re.match(str(h))
    if not m:
        return snake(h), None, str(h)
    base = m.group("base")
    unit = m.group("u1") or m.group("u2")
    return snake(base), (unit.strip() if unit else None), str(h)

class MapSpec:
    def __init__(self, path, data_key, map_key=None, suffix=""):
        self.path = Path(path)
        self.data_key = data_key
        self.map_key = map_key
        self.suffix = suffix

def parse_map_specs(values):
    specs = []
    for raw in values:
        cfg = {}
        for p in raw.split(";"):
            if p and "=" in p:
                k, v = p.split("=", 1)
                cfg[k.strip()] = v.strip()
        if "path" not in cfg or "data_key" not in cfg:
            raise ValueError("map-spec requires path=...;data_key=...")
        specs.append(MapSpec(cfg["path"], cfg["data_key"], cfg.get("map_key"), cfg.get("suffix", "")))
    return specs

def read_map_table(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype="object")
    parsed = [parse_header(c) for c in df.columns]
    cols = [p[0] for p in parsed]
    df.columns = cols
    units = {p[0]: (p[1] or "") for p in parsed}
    originals = {p[0]: p[2] for p in parsed}
    return df, units, originals

def apply_maps_to_series(series_list, specs):
    for spec in specs:
        mdf, _, _ = read_map_table(spec.path)
        right_key = spec.map_key or spec.data_key
        if right_key not in mdf.columns:
            raise KeyError(f"map_key '{right_key}' not found in {spec.path}")
        cols_to_add = [c for c in mdf.columns if c != right_key]
        records = {}
        for i in range(len(mdf)):
            k = mdf[right_key].iloc[i]
            records[k] = {c + spec.suffix: mdf[c].iloc[i] for c in cols_to_add}
        for s in series_list:
            if hasattr(s, spec.data_key):
                key = getattr(s, spec.data_key)
                if key in records:
                    for k, v in records[key].items():
                        setattr(s, k, v)

def series_from_file(path, name_from=None):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype="object")
    parsed = [parse_header(c) for c in df.columns]
    cols = [p[0] for p in parsed]
    units = {p[0]: (p[1] or "") for p in parsed}
    # originals = {p[0]: p[2] for p in parsed}
    df.columns = cols
    params = {}
    for c in cols:
        col = df[c]
        col_num = pd.to_numeric(col, errors="coerce")
        if col.nunique(dropna=False) == 1:
            params[c] = col.iloc[0]
        else:
            if not col_num.isna().all():
                arr = col_num.to_numpy(dtype=float)
            else:
                arr = col.to_numpy()
            params[c] = arr
    params["units"] = units
    # params["original_headers"] = originals
    params["source_file"] = str(path)
    if name_from and name_from in params and not isinstance(params[name_from], np.ndarray):
        params["name"] = str(params[name_from])
    else:
        params["name"] = path.stem

    params["title"] = params["name"]
    params["filename"] = params["name"]
    return Series(params)

def load_data(data_dir, maps=None, recursive=True, exts=("csv","tsv","dat","txt"), name_from=None):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Not found: {data_dir}")
    patterns = [f"**/*.{str(e).lstrip('.').lower()}" for e in exts]
    files = []
    for p in patterns:
        files.extend(sorted(data_dir.glob(p)))
    if not recursive:
        files = [f for f in files if f.parent == data_dir]
    out = []
    for f in files:
        try:
            s = series_from_file(f, name_from=name_from)
            out.append(s)
        except Exception as e:
            print(f"[WARN] Skipping {f}: {e}", file=sys.stderr)
    if maps:
        specs = parse_map_specs(maps)
        apply_maps_to_series(out, specs)
    return out
