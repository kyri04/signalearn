from copy import deepcopy

import numpy as np

from .. import neural

batched = True
persistence_name = "torch"
persistence_format = "state_dict"
persistence_payload = "model.pt"
persistence_dependencies = ("torch", "numpy", "dask")

fitted_inputs = neural.fitted_inputs
prediction_schema = neural.prediction_schema
output_schema = neural.output_schema


def framework():
    try:
        import torch
    except ImportError as error:
        raise ImportError("Install signalearn[torch] to use Model") from error
    return torch


def supports(model, operation):
    from ...components import Model
    from ..models import native_framework

    return isinstance(model, Model) and native_framework(model.model) == "torch"


def snapshot(component):
    torch = framework()
    if not isinstance(component.model, torch.nn.Module):
        raise TypeError("Model requires a torch.nn.Module instance")
    return deepcopy(component)


def tensor_batch(component, batch):
    torch = framework()
    inputs, labels, parameters = batch
    inputs = {name: torch.as_tensor(value, device=component.device) for name, value in inputs.items()}
    if labels is not None:
        if component.task != "classification":
            dtype = next(component.model.parameters(), torch.empty((), dtype=torch.float64)).dtype
            labels = labels.astype(torch.empty((), dtype=dtype).numpy().dtype)
        labels = torch.as_tensor(labels, device=component.device)
    parameters = {name: torch.as_tensor(value, device=component.device) if isinstance(value, np.ndarray) else value
                  for name, value in parameters.items()}
    return inputs, labels, parameters


def forward(component, inputs, parameters):
    if len(inputs) == 1:
        return component.model(next(iter(inputs.values())), **parameters)
    return component.model(**inputs, **parameters)


class TensorBatches:
    def __init__(self, component, batches):
        self.component = component
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        for batch in self.batches:
            yield tensor_batch(self.component, batch)

    def shuffle(self):
        self.batches.shuffle()


def fit_fields(component, fields, names, target, parameters, ids):
    torch = framework()
    fields, names, target, parameters = neural.fit_context(component, fields, names, target, parameters, ids)
    fitted = snapshot(component)
    initial = getattr(component, "_initial_weights", component.model.state_dict())
    fitted.model.load_state_dict(deepcopy(initial))
    fitted._initial_weights = {name: value.detach().cpu().clone() for name, value in initial.items()}
    fitted.model.to(fitted.device)
    neural.record_fit(fitted, fields, names, target)
    batches = TensorBatches(fitted, neural.Batches(fitted, fields, names, target, parameters))
    if fitted.trainer is not None:
        fitted.trainer(fitted, batches)
    else:
        if not callable(fitted.loss) or not callable(fitted.optimizer):
            raise TypeError("Torch training requires a loss and an optimizer factory, or a custom trainer")
        optimizer = fitted.optimizer(fitted.model.parameters())
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must construct a torch.optim.Optimizer for the copied model")
        native_parameters = {id(parameter) for parameter in fitted.model.parameters()}
        if any(id(parameter) not in native_parameters for group in optimizer.param_groups for parameter in group["params"]):
            raise ValueError("optimizer must be bound to the copied model's parameters")
        if isinstance(fitted.loss, torch.nn.Module):
            fitted.loss.to(fitted.device)
        fitted.model.train()
        for epoch in range(fitted.epochs):
            if fitted.shuffle:
                batches.shuffle()
            for inputs, labels, options in batches:
                optimizer.zero_grad()
                prediction = forward(fitted, inputs, options)
                if fitted.task != "classification" and prediction.shape == (len(labels), 1):
                    labels = labels[:, None]
                loss = fitted.loss(prediction, labels)
                if loss.ndim != 0 or not torch.isfinite(loss):
                    raise ValueError("Training loss must be a finite scalar")
                loss.backward()
                optimizer.step()
    fitted.model.eval()
    return fitted


def numpy_output(value):
    if isinstance(value, dict):
        return {name: numpy_output(item) for name, item in value.items()}
    torch = framework()
    if not isinstance(value, torch.Tensor):
        raise TypeError("Native Torch outputs must be tensors or a mapping of tensors")
    return value.detach().cpu().numpy()


def apply_batch(component, reader, indices, schema, output):
    torch = framework()
    local = snapshot(component)
    local.model.to(local.device).eval()
    inputs, _, parameters = tensor_batch(local, reader.read(indices))
    with torch.no_grad():
        raw = numpy_output(forward(local, inputs, parameters))
    if output == "transform":
        return neural.validated_transform(raw, schema, indices)
    attributes = {name: entry["attributes"] for name, entry in schema.items()}
    return neural.prediction_result(local, raw, attributes, output, len(indices))


def deferred_output(*args):
    import sys

    return neural.deferred_output(sys.modules[__name__], *args)


def validate_persistence(component):
    from ..neural_io import validate_persistence

    validate_persistence(component, "torch")


def persistence_metadata(component):
    from .sklearn import persistence_metadata

    return persistence_metadata(component)


def dump_model(component, path):
    from ..neural_io import write_configuration

    write_configuration(component, path.parent, "torch")
    framework().save({name: value.detach().cpu() for name, value in component.model.state_dict().items()}, path)


def load_model(path):
    from ..neural_io import read_configuration

    component = read_configuration(path.parent, "torch")
    component.model.load_state_dict(framework().load(path, map_location="cpu", weights_only=True))
    component.model.eval()
    return component


def restore_metadata(component, metadata):
    from ..neural_io import restore_metadata

    restore_metadata(component, metadata)
