import re, sys
from pathlib import Path
import pandas as pd
import numpy as np
from signalearn.classes import Series
from signalearn.general_utility import snake
import glob
import os

from pathlib import Path
import re
from typing import Dict, Any, List, Union

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

def series_from_file(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", dtype="object")
    parsed = [parse_header(c) for c in df.columns]
    cols = [p[0] for p in parsed]
    units = {p[0]: (p[1] or "") for p in parsed}
    labels = {p[0]: p[0] for p in parsed}
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
    params["labels"] = labels
    # params["source_file"] = str(path)
    params["name"] = path.stem

    # params["title"] = params["name"]
    # params["filename"] = params["name"]
    return Series(params)

def _load_map(map_source):
    if isinstance(map_source, dict):
        sample_key = next(iter(map_source))
        sample_row = map_source[sample_key]
        key_name = next(iter(sample_row)) if False else "id_scan"
        return key_name, map_source
    p = Path(map_source)
    with p.open("r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n\r")
        if "\t" in header:
            delim = "\t"
        elif "," in header:
            delim = ","
        else:
            delim = None
        cols = header.split(delim) if delim else header.split()
        key_col = cols[0]
        data: Dict[str, Dict[str, Any]] = {}
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(delim) if delim else line.split()
            if len(parts) < len(cols):
                continue
            row = dict(zip(cols, parts))
            k = row[key_col]
            data[k] = row
    return key_col, data

def load_data(directory,
              map = None,
              spec = None):
    
    exts = (".dat", ".txt", ".csv")
    directory = Path(directory)
    map_key = None
    map_data: Dict[str, Dict[str, Any]] | None = None
    if map is not None:
        map_key, map_data = _load_map(map)
    out = []
    for ext in exts:
        for fp in directory.glob(f"*{ext}"):
            series = series_from_file(fp)
            if spec:
                attrs = extract_attrs_from_filename(fp, spec)
                for k, v in attrs.items():
                    setattr(series, k, v)
            if map_data is not None:
                ident = getattr(series, map_key, None)
                if ident is None and spec:
                    attrs2 = extract_attrs_from_filename(fp, spec)
                    ident = attrs2.get(map_key)
                if ident is not None:
                    row = map_data.get(str(ident))
                    if row:
                        for k, v in row.items():
                            setattr(series, k, v)
            out.append(series)
    return out

def add_header(directory, header_cols):
    exts = ("*.dat", "*.txt", "*.csv")
    delimiters = [",", "\t", ";", " "]

    def detect_delim(line):
        scores = {d: line.count(d) for d in delimiters}
        return max(scores, key=scores.get) if max(scores.values()) > 0 else ","

    for ext in exts:
        for file in glob.glob(os.path.join(directory, ext)):
            with open(file, "r") as f:
                lines = f.readlines()

            first_data = next((l for l in lines if l.strip()), "")
            delim = detect_delim(first_data)
            header = delim.join(header_cols) + "\n"

            with open(file, "w") as f:
                f.write(header + "".join(lines))

def extract_attrs_from_filename(path, spec):
    p = Path(path)
    stem = p.stem
    split_re = spec.get("split", r"[-_]+")
    tokens = re.split(split_re, stem)
    fields = spec.get("fields", {})

    attrs: Dict[str, Any] = {}
    for field, recipe in fields.items():
        if isinstance(recipe, int):
            idx = _norm_index(recipe, len(tokens))
            attrs[field] = tokens[idx]
        elif isinstance(recipe, list):
            parts: List[str] = []
            for r in recipe:
                if isinstance(r, int):
                    idx = _norm_index(r, len(tokens))
                    parts.append(tokens[idx])
                elif isinstance(r, str):
                    parts.append(r)

            attrs[field] = "".join(parts)

    if "transform" in spec:
        attrs = _apply_transforms(attrs, spec["transform"])
    return attrs

def _norm_index(i, n):
    if i < 0:
        i = n + i

    return i

def _apply_transforms(attrs, tf):
    out = dict(attrs)
    for k, cfg in tf.items():
        if k not in out:
            continue
        v = out[k]
        funcs = cfg if isinstance(cfg, (list, tuple)) else [cfg]
        for f in funcs:
            v = f(v)
        out[k] = v
    return out