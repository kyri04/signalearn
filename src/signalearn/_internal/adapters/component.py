from collections.abc import Mapping
from copy import deepcopy

import numpy as np

from ..metadata import DISPLAY_ATTRIBUTES
from ..tensor import CoordinateReference, check_shape, dimensions

structured = True


def supports(model):
    return not isinstance(model, type) and hasattr(model, "signalearn_mode")


def fitted_inputs(model, fields, names):
    if model.signalearn_mode not in {"sample", "batch"}:
        raise ValueError("signalearn_mode must be 'sample' or 'batch'")
    if not callable(getattr(model, "transform", None)) or not callable(getattr(model, "signalearn_outputs", None)):
        raise TypeError("Custom components require transform and signalearn_outputs methods")
    declarations = getattr(model, "signalearn_inputs", None)
    if not isinstance(declarations, Mapping) or not declarations or set(declarations) != set(names):
        raise ValueError("signalearn_inputs must declare exactly the selected input field names")
    lookup = dict(zip(names, fields))
    for name, declaration in declarations.items():
        if not isinstance(declaration, Mapping) or "dims" not in declaration:
            raise ValueError(f"Input {name!r} must declare its dims")
        if set(declaration) - {"dims", "dtype", "unit", "shape"}:
            raise ValueError(f"Input {name!r} has unsupported declaration attributes")
        dims, shape = dimensions(declaration["dims"], declaration.get("shape"))
        field = lookup[name]
        if dims != field.dims:
            raise ValueError(f"Input {name!r} requires dimensions {dims!r}, got {field.dims!r}")
        for attribute in ("dtype", "unit"):
            if attribute in declaration and declaration[attribute] != getattr(field, attribute):
                raise ValueError(f"Input {name!r} requires {attribute} {declaration[attribute]!r}; no conversion is inferred")
        if dims:
            for item in field._values.items:
                check_shape(item.shape, dims, shape, f"Input {name!r}")
    names = tuple(declarations)
    return tuple(lookup[name] for name in names), names


def field_metadata(field):
    shapes = tuple(tuple(item.shape) for item in field._values.items) if field.dims else ((),) * len(field)
    return {
        "dtype": field.dtype, "dims": field.dims, "shapes": shapes,
        "shape": shapes[0] if shapes and all(shape == shapes[0] for shape in shapes) else None,
        "nullable": field.nullable,
        "coords": {dim: {"field": value.name} if isinstance(value, CoordinateReference) else list(value)
                   for dim, value in field._coords.items()},
        **{key: dict(getattr(field, key)) if key in {"colours", "markers"} else getattr(field, key)
           for key in DISPLAY_ATTRIBUTES},
    }


def output_schema(model, fields, names):
    from ...data import Field

    inputs = dict(zip(names, fields))
    declarations = model.signalearn_outputs({name: field_metadata(field) for name, field in inputs.items()})
    if not isinstance(declarations, Mapping) or not declarations:
        raise ValueError("signalearn_outputs must return a non-empty mapping of output field declarations")
    schema = {}
    for name, declaration in declarations.items():
        if not isinstance(name, str) or not name or name == "id" or not isinstance(declaration, Mapping):
            raise ValueError("Outputs require non-empty field names other than 'id' and metadata mappings")
        if set(declaration) - {"source", "dtype", "dims", "shape", "sample_shapes", "coords", "nullable", "prediction", *DISPLAY_ATTRIBUTES}:
            raise ValueError(f"Output {name!r} has unsupported declaration attributes")
        declaration = deepcopy(dict(declaration))
        source_name = declaration.pop("source", None)
        if source_name is not None and source_name not in inputs:
            raise ValueError(f"Output {name!r} references unknown source {source_name!r}")
        source = inputs.get(source_name)
        attributes = field_metadata(source) if source is not None else {}
        attributes.pop("shapes", None)
        attributes.pop("shape", None)
        attributes.update(declaration)
        sample_shapes = attributes.pop("sample_shapes", None)
        if "dtype" not in attributes:
            raise ValueError(f"Output {name!r} must declare dtype or a source")
        dims, shape = dimensions(attributes.get("dims", ()), attributes.get("shape"))
        attributes["dims"] = dims
        if source is not None and dims != source.dims and "coords" not in declaration:
            attributes["coords"] = {}
        length = len(fields[0])
        if not dims:
            if sample_shapes is not None:
                raise ValueError("Scalar outputs do not take sample_shapes")
            shapes = ((),) * length
        elif sample_shapes is not None:
            if not isinstance(sample_shapes, (tuple, list)) or len(sample_shapes) != length:
                raise ValueError(f"Output {name!r} needs one sample_shapes entry per input sample")
            shapes = tuple(dimensions(dims, item)[1] for item in sample_shapes)
        elif shape is not None and all(size is not None for size in shape):
            shapes = (shape,) * length
        elif source is not None and dims == source.dims:
            shapes = tuple(tuple(item.shape) for item in source._values.items)
        else:
            raise ValueError(f"Output {name!r} must declare a fixed shape, sample_shapes, or an unchanged source structure")
        for actual in shapes:
            if actual is None or any(size is None for size in actual):
                raise ValueError(f"Output {name!r} requires known per-sample shapes")
            check_shape(actual, dims, shape, f"Output {name!r}")
        placeholder = Field([], ids=(), **attributes)
        schema[name] = {"attributes": attributes, "shapes": shapes, "source": source,
                        "dtype": placeholder.dtype, "nullable": placeholder.nullable}
    return schema


def snapshot(model):
    return deepcopy(model)


def apply_transform(model, inputs, parameters, schema, shapes, batch):
    from ..data import validate_block
    from ..tensor import validate_array

    model = deepcopy(model)
    result = model.transform(inputs, **parameters)
    if not isinstance(result, Mapping) or set(result) != set(schema):
        raise ValueError("Component transform must return exactly the declared output field names")
    validated = {}
    for name, specification in schema.items():
        value = result[name]
        dims = specification["dims"]
        expected = shapes[name]
        context = f"Component {type(model).__name__}, output {name!r}"
        if batch:
            if (not isinstance(value, (np.ndarray, tuple, list)) or (isinstance(value, np.ndarray) and value.ndim == 0)
                    or len(value) != len(expected)):
                raise ValueError(f"{context}: batch output must preserve sample count and order")
            if dims:
                validated[name] = tuple(validate_array(item, specification["dtype"], specification["nullable"], shape, context)
                                        for item, shape in zip(value, expected))
            else:
                if np.ma.isMaskedArray(value):
                    raise ValueError(f"{context}: masked outputs require an explicit supported representation")
                validated[name] = validate_block(value, logical_dtype=specification["dtype"], nullable=specification["nullable"])
        elif dims:
            validated[name] = validate_array(value, specification["dtype"], specification["nullable"], expected, context)
        else:
            if np.ma.isMaskedArray(value) or np.ndim(value) != 0:
                raise ValueError(f"{context}: expected one scalar value")
            value = np.asarray(value).item()
            validated[name] = validate_block([value], logical_dtype=specification["dtype"], nullable=specification["nullable"])[0]
    return validated
