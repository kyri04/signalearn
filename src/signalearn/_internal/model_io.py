from collections.abc import Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, packages_distributions, version
import json
from pathlib import Path
import platform

import numpy as np

from .io import reject_constant, unique_object
from .metadata import DISPLAY_ATTRIBUTES, DTYPES, prepare_colours, prepare_markers, validate_metadata


def importable_reference(value):
    module, name = getattr(value, "__module__", None), getattr(value, "__qualname__", None)
    if not isinstance(module, str) or not isinstance(name, str) or module == "__main__" or "<locals>" in name:
        raise TypeError("Custom classes and functions must be defined in an importable module, not a notebook, "
                        "lambda, or local scope; put the definition in your workspace extension package")
    try:
        imported = import_module(module)
        for part in name.split("."):
            imported = getattr(imported, part)
    except (ImportError, AttributeError) as error:
        raise TypeError(f"Cannot import {module}.{name}; custom definitions must be available as module attributes") from error
    if imported is not value:
        raise TypeError(f"{module}.{name} does not resolve to the component's current definition")
    return {"module": module, "qualname": name}


def target_metadata(target, *, encoded):
    if target is None:
        return None
    required = {"name", "dtype", *DISPLAY_ATTRIBUTES}
    if not isinstance(target, Mapping) or set(target) != required:
        raise ValueError("Model target metadata has invalid fields")
    target = dict(target)
    if (not isinstance(target["name"], str) or not target["name"]
            or not isinstance(target["dtype"], str) or target["dtype"] not in DTYPES):
        raise ValueError("Model target metadata requires a name and a supported scalar dtype")
    validate_metadata(target["dtype"], target["unit"], target["label"], False,
                      target["unit_latex"], target["label_latex"], target["colour"], target["marker"])
    for attribute, prepare in (("colours", prepare_colours), ("markers", prepare_markers)):
        values = target[attribute]
        if encoded:
            if not isinstance(values, list) or any(not isinstance(item, list) or len(item) != 2 for item in values):
                raise ValueError(f"Model {attribute} must be a list of category/display pairs")
            mapping = {}
            for category, display in values:
                if not isinstance(category, (str, bool, int, float)):
                    raise ValueError(f"Model {attribute} contains an invalid category")
                if category in mapping:
                    raise ValueError(f"Model {attribute} contains duplicate categories")
                mapping[category] = display
        else:
            mapping = values
        target[attribute] = dict(prepare(mapping, target["dtype"]))
    return target


def validate_model_metadata(metadata, *, encoded=False):
    if not isinstance(metadata, Mapping) or set(metadata) != {"inputs", "target"}:
        raise ValueError("Model metadata requires inputs and target entries")
    inputs = metadata["inputs"]
    if inputs is not None:
        if not isinstance(inputs, list) or not inputs:
            raise ValueError("Model inputs must be a non-empty list or null")
        names = set()
        for field in inputs:
            if (not isinstance(field, Mapping) or not {"name", "kind", "unit"} <= field.keys()
                    or field.keys() - {"name", "kind", "unit", "vector"}):
                raise ValueError("Each model input requires name, kind, and unit")
            if not isinstance(field["name"], str) or not field["name"] or field["name"] in names:
                raise ValueError("Model input field names must be non-empty and unique")
            if not isinstance(field["kind"], str) or field["kind"] not in {"numeric", "string", "bool"}:
                raise ValueError("Model input logical kind is unsupported")
            if field["unit"] is not None and not isinstance(field["unit"], str):
                raise ValueError("Model input units must be strings or null")
            if "vector" in field:
                validate_vector_metadata(field["vector"])
            names.add(field["name"])
        inputs = [dict(field) for field in inputs]
    return {"inputs": inputs, "target": target_metadata(metadata["target"], encoded=encoded)}


def validate_vector_metadata(vector):
    from .data import validate_block

    if not isinstance(vector, Mapping) or set(vector) != {"dim", "size", "coords"}:
        raise ValueError("Vector feature metadata requires dim, size, and coords")
    if not isinstance(vector["dim"], str) or not vector["dim"] or type(vector["size"]) is not int or vector["size"] < 1:
        raise ValueError("Vector feature dimensions and sizes are invalid")
    coords = vector["coords"]
    if coords is not None:
        if not isinstance(coords, Mapping) or set(coords) != {"values", "dtype", "unit"}:
            raise ValueError("Vector feature coordinates require values, dtype, and unit")
        if not isinstance(coords["values"], list) or len(coords["values"]) != vector["size"]:
            raise ValueError("Vector feature coordinates must match the feature width")
        if (not isinstance(coords["dtype"], str) or coords["dtype"] not in DTYPES
                or coords["unit"] is not None and not isinstance(coords["unit"], str)):
            raise ValueError("Vector feature coordinate types or units are invalid")
        validate_block(np.asarray(coords["values"], dtype=object), logical_dtype=coords["dtype"], nullable=False)


def encoded_metadata(metadata):
    result = validate_model_metadata(metadata)
    if result["target"] is not None:
        for attribute in ("colours", "markers"):
            result["target"][attribute] = [
                [key.item() if isinstance(key, np.generic) else key, value]
                for key, value in result["target"][attribute].items()
            ]
    return result


def model_manifest(model, adapter, metadata):
    component = importable_reference(type(model))
    dependencies = set(adapter.persistence_dependencies)
    root = component["module"].split(".")[0]
    if root != "sklearn":
        dependencies.update(packages_distributions().get(root, ()))
    versions = {name: version(name) for name in sorted(dependencies)}
    versions.update(python=platform.python_version(), signalearn=version("signalearn"))
    return {"version": 1, "kind": "signalearn.model", "adapter": adapter.persistence_name,
            "format": adapter.persistence_format, "payload": adapter.persistence_payload,
            "component": component, "versions": versions, "metadata": encoded_metadata(metadata)}


def write_model_manifest(directory, manifest):
    (directory / "model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )


def read_model_manifest(path):
    directory = Path(path).expanduser().absolute()
    if not directory.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {directory}")
    try:
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"),
                              object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, UnicodeError) as error:
        raise ValueError(f"{directory / 'model.json'}: invalid model manifest: {error}") from error
    required = {"version", "kind", "adapter", "format", "payload", "component", "versions", "metadata"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("Model manifest has invalid fields")
    if type(manifest["version"]) is not int or manifest["version"] != 1 or manifest["kind"] != "signalearn.model":
        raise ValueError("Unsupported model artifact kind or format version")
    if not isinstance(manifest["adapter"], str):
        raise ValueError("Model adapter must be a string")
    component = manifest["component"]
    if (not isinstance(component, dict) or set(component) != {"module", "qualname"}
            or any(not isinstance(value, str) or not value or not all(part.isidentifier() for part in value.split('.'))
                   for value in component.values()) or component["module"] == "__main__"):
        raise ValueError("Model component reference is invalid")
    return directory, manifest


def model_payload(directory, manifest, adapter):
    if manifest["format"] != adapter.persistence_format or manifest["payload"] != adapter.persistence_payload:
        raise ValueError("Model payload or serialization format does not match its adapter")
    path = directory / adapter.persistence_payload
    if not path.is_file():
        raise FileNotFoundError(f"Model payload does not exist: {path}")
    return path


def check_model_versions(versions, adapter):
    required = {"python", "signalearn", *adapter.persistence_dependencies}
    if (not isinstance(versions, dict) or not required <= versions.keys()
            or any(not isinstance(name, str) or not name or not isinstance(saved, str) or not saved
                   for name, saved in versions.items())):
        raise ValueError("Model dependency versions are invalid or incomplete")
    if versions["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        raise RuntimeError(f"Model requires Python {versions['python']}; current version is {platform.python_version()}")
    for name, saved in versions.items():
        if name in {"python", "signalearn"}:
            continue
        try:
            current = version(name)
        except PackageNotFoundError as error:
            raise ImportError(f"Model requires {name}=={saved}; install the saved model's dependencies") from error
        if current != saved:
            raise RuntimeError(f"Model requires {name}=={saved}; current version is {current}. "
                               "Restore a compatible environment before loading")


def validate_restored_model(model, manifest, adapter):
    actual = {"module": type(model).__module__, "qualname": type(model).__qualname__}
    if actual != manifest["component"]:
        raise ValueError("Loaded component class does not match the model manifest")
    if not adapter.supports(model, "save_model"):
        raise TypeError("Loaded object is not supported by its persistence adapter")
    adapter.validate_persistence(model)
