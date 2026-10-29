from importlib import import_module
import json

from .io import reject_constant, unique_object
from .model_io import importable_reference
from .neural import validate_component
from .prediction import class_values


def configuration(component, framework):
    result = {name: getattr(component, name) for name in (
        "inputs", "task", "output", "outputs", "batch_size", "device",
    )}
    result["classes"] = class_values(component.classes)[0] if component.task == "classification" else None
    result["input_signature"] = getattr(component, "_input_signature", None)
    result["fitted"] = getattr(component, "_fitted", False)
    if framework == "torch":
        if not callable(component.factory):
            raise TypeError("Saving Model requires an importable factory and JSON factory_params for its architecture")
        result["factory"] = importable_reference(component.factory)
        result["factory_params"] = dict(component.factory_params or {})
    return result


def validate_persistence(component, framework):
    validate_component(component)
    if component.task != "transform" and not getattr(component, "_fitted", False):
        raise ValueError("Fit the component before saving")
    try:
        json.dumps(configuration(component, framework), allow_nan=False)
    except (TypeError, ValueError) as error:
        error.add_note("Component declarations and architecture parameters must be JSON serializable")
        raise


def write_configuration(component, directory, framework):
    (directory / "component.json").write_text(
        json.dumps(configuration(component, framework), allow_nan=False, indent=2) + "\n", encoding="utf-8",
    )


def restore_metadata(component, metadata):
    from .adapters.sklearn import restore_metadata as restore

    signature = getattr(component, "_input_signature", None)
    if signature is not None:
        expected = [{"name": name, "kind": "bool" if item["dtype"] == "bool" else "numeric", "unit": item["unit"]}
                    for name, item in signature.items()]
        if metadata["inputs"] != expected:
            raise ValueError("Model manifest inputs do not match the neural component declaration")
    restore(component, metadata)


def read_configuration(directory, framework, model=None):
    from ..components import Model
    from .adapters import torch

    options = json.loads((directory / "component.json").read_text(encoding="utf-8"),
                         object_pairs_hook=unique_object, parse_constant=reject_constant)
    required = {"inputs", "task", "output", "outputs", "classes", "batch_size", "device", "input_signature", "fitted"}
    if framework == "torch":
        required |= {"factory", "factory_params"}
    if not isinstance(options, dict) or set(options) != required:
        raise ValueError("Invalid neural component configuration")
    signature, fitted = options.pop("input_signature"), options.pop("fitted")
    if type(fitted) is not bool:
        raise ValueError("Invalid fitted state")
    options["device"] = "cpu" if framework == "torch" else "/CPU:0"
    if framework == "torch":
        reference = options["factory"]
        if not isinstance(reference, dict) or set(reference) != {"module", "qualname"}:
            raise ValueError("Invalid architecture factory reference")
        factory = import_module(reference["module"])
        for part in reference["qualname"].split("."):
            factory = getattr(factory, part)
        importable_reference(factory)
        options["factory"] = factory
        model = factory(**options["factory_params"])
        if not isinstance(model, torch.framework().nn.Module):
            raise TypeError("Saved factory must construct a torch.nn.Module")
    component = Model(model, **options)
    component._fitted = fitted
    if signature is not None:
        if not isinstance(signature, dict) or set(signature) != set(component.inputs):
            raise ValueError("Invalid fitted input signature")
        for entry in signature.values():
            if not isinstance(entry, dict) or set(entry) != {"dtype", "dims", "shape", "unit", "coords"}:
                raise ValueError("Invalid fitted input declaration")
            entry["dims"], entry["shape"] = tuple(entry["dims"]), tuple(entry["shape"])
        component._input_signature = signature
    validate_component(component)
    return component
