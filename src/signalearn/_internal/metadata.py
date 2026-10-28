from collections.abc import Mapping
from numbers import Integral, Real
from types import MappingProxyType
import math

from matplotlib.colors import is_color_like
from matplotlib.markers import MarkerStyle

DTYPES = frozenset({"string", "bool", "int64", "float64"})
ARRAY_DTYPES = frozenset({"bool", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32",
                          "uint64", "float16", "float32", "float64"})
DISPLAY_ATTRIBUTES = ("unit", "label", "unit_latex", "label_latex", "colour", "colours", "marker", "markers")


def validate_colour(colour):
    if not isinstance(colour, str):
        raise TypeError("A colour must be a Matplotlib colour string")
    if not is_color_like(colour):
        raise ValueError(f"Invalid Matplotlib colour: {colour!r}")


def validate_marker(marker):
    if not isinstance(marker, str):
        raise TypeError("A marker must be a Matplotlib marker string")
    try:
        MarkerStyle(marker)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid Matplotlib marker: {marker!r}") from error


def validate_metadata(dtype, unit, label, nullable, unit_latex=None, label_latex=None, colour=None, marker=None,
                      *, array=False):
    supported = ARRAY_DTYPES if array else DTYPES
    if dtype is not None and (not isinstance(dtype, str) or dtype not in supported):
        raise ValueError(f"dtype must be one of {sorted(supported)}")
    for name, value in (("unit", unit), ("label", label),
                        ("unit_latex", unit_latex), ("label_latex", label_latex)):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{name} must be a string or None")
    if nullable is not None and not isinstance(nullable, bool):
        raise TypeError("nullable must be a boolean or None")
    if colour is not None:
        validate_colour(colour)
    if marker is not None:
        validate_marker(marker)


def prepare_colours(colours, dtype):
    return prepare_category_display(colours, dtype, "colours", validate_colour)


def prepare_markers(markers, dtype):
    return prepare_category_display(markers, dtype, "markers", validate_marker)


def prepare_category_display(mapping, dtype, attribute, validate):
    if mapping is None:
        return MappingProxyType({})
    if not isinstance(mapping, Mapping):
        raise TypeError(f"{attribute} must map category values to Matplotlib display strings")
    result = {}
    for category, value in mapping.items():
        if dtype == "string":
            valid = isinstance(category, str)
        elif dtype == "bool":
            valid = isinstance(category, bool)
        elif dtype.startswith(("int", "uint")):
            valid = isinstance(category, Integral) and not isinstance(category, bool)
        else:
            valid = isinstance(category, Real) and not isinstance(category, bool)
            if valid:
                try:
                    valid = math.isfinite(category)
                except OverflowError:
                    valid = False
        if not valid:
            raise ValueError(f"{attribute} category {category!r} is incompatible with {dtype}")
        validate(value)
        result[category] = value
    return MappingProxyType(result)


def updated_metadata(field, changes):
    if not isinstance(changes, Mapping):
        raise TypeError("Each field update must be a mapping of metadata attributes")
    unknown = set(changes) - set(DISPLAY_ATTRIBUTES)
    if unknown:
        raise ValueError(f"Unsupported metadata attributes: {sorted(unknown, key=repr)!r}")
    result = {name: getattr(field, name) for name in DISPLAY_ATTRIBUTES}
    result.update(changes)
    validate_metadata(field.dtype, result["unit"], result["label"], field.nullable,
                      result["unit_latex"], result["label_latex"], result["colour"], result["marker"],
                      array=bool(field.dims))
    for name, prepare in (("colours", prepare_colours), ("markers", prepare_markers)):
        if name not in changes:
            continue
        patch = changes[name]
        if patch is not None:
            if not isinstance(patch, Mapping):
                raise TypeError(f"{name} must be a category-to-display mapping or None")
            patch = {**getattr(field, name), **patch}
        result[name] = prepare(patch, field.dtype)
    return result
