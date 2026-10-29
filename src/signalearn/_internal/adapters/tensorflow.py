from copy import deepcopy

from .. import neural

batched = True
persistence_name = "tensorflow"
persistence_format = "keras"
persistence_payload = "model.keras"
persistence_dependencies = ("tensorflow", "keras", "numpy", "dask")

fitted_inputs = neural.fitted_inputs
prediction_schema = neural.prediction_schema
output_schema = neural.output_schema


def framework():
    try:
        import tensorflow as tf
        import keras
    except ImportError as error:
        raise ImportError("Install signalearn[tensorflow] to use Model") from error
    if keras.backend.backend() != "tensorflow":
        raise ValueError("Model currently requires the TensorFlow backend")
    return tf, keras


def supports(model, operation):
    from ...components import Model
    from ..models import native_framework

    return isinstance(model, Model) and native_framework(model.model) == "tensorflow"


def snapshot(component):
    _, keras = framework()
    if not isinstance(component.model, keras.Model) or not component.model.built:
        raise TypeError("Model requires a built native Keras model")
    copied = deepcopy({name: value for name, value in vars(component).items() if name != "model"})
    result = object.__new__(type(component))
    result.__dict__.update(copied)
    result.model = keras.models.clone_model(component.model)
    result.model.set_weights(component.model.get_weights())
    config = component.model.get_compile_config()
    if config:
        result.model.compile_from_config(config)
    return result


def native_inputs(inputs):
    return next(iter(inputs.values())) if len(inputs) == 1 else inputs


def fit_fields(component, fields, names, target, parameters, ids):
    tf, keras = framework()
    fields, names, target, parameters = neural.fit_context(component, fields, names, target, parameters, ids)
    fitted = snapshot(component)
    initial = getattr(component, "_initial_weights", component.model.get_weights())
    fitted.model.set_weights(initial)
    fitted._initial_weights = deepcopy(initial)
    neural.record_fit(fitted, fields, names, target)
    reader = neural.Batches(fitted, fields, names, target, parameters if fitted.trainer is not None else None)
    if fitted.trainer is not None:
        with tf.device(fitted.device):
            fitted.trainer(fitted, reader)
    else:
        if any(hasattr(value, "ids") for value in parameters.values()):
            raise TypeError("Field-valued Keras fit parameters require a custom trainer")
        if set(parameters) & {"epochs", "batch_size", "shuffle", "validation_split", "validation_data", "x", "y"}:
            raise ValueError("Configure epochs, batch_size, and shuffle on Model; use a trainer for validation")
        if not getattr(fitted.model, "optimizer", None):
            raise ValueError("Compile the native Keras model before fitting, or supply a custom trainer")
        loss = fitted.model.loss
        if fitted.task == "classification" and "sparse_categorical_crossentropy" not in str(
                keras.losses.serialize(keras.losses.get(loss))).lower():
            raise ValueError("Standard classification uses encoded class indices; compile with SparseCategoricalCrossentropy")
        if fitted.task == "classification" and bool(getattr(keras.losses.get(loss), "from_logits", False)) != (
                fitted.output == "logits"):
            raise ValueError("Keras loss from_logits must match the declared output")

        class Dataset(keras.utils.PyDataset):
            def __init__(self):
                super().__init__(workers=1, use_multiprocessing=False, max_queue_size=1)
                if fitted.shuffle:
                    reader.shuffle()

            def __len__(self):
                return len(reader)

            def __getitem__(self, index):
                inputs, labels, _ = reader[index]
                return native_inputs(inputs), labels

            def on_epoch_end(self):
                if fitted.shuffle:
                    reader.shuffle()

        with tf.device(fitted.device):
            fitted.model.fit(Dataset(), epochs=fitted.epochs, shuffle=False, **{"verbose": 0, **parameters})
    return fitted


def apply_batch(component, reader, indices, schema, output):
    tf, _ = framework()
    local = snapshot(component)
    inputs, _, parameters = reader.read(indices)
    with tf.device(local.device):
        raw = local.model(native_inputs(inputs), training=False, **parameters)
        raw = tf.nest.map_structure(lambda value: value.numpy(), raw)
    if output == "transform":
        return neural.validated_transform(raw, schema, indices)
    attributes = {name: entry["attributes"] for name, entry in schema.items()}
    return neural.prediction_result(local, raw, attributes, output, len(indices))


def deferred_output(*args):
    import sys

    return neural.deferred_output(sys.modules[__name__], *args)


def validate_persistence(component):
    from ..neural_io import validate_persistence

    validate_persistence(component, "tensorflow")


def persistence_metadata(component):
    from .sklearn import persistence_metadata

    return persistence_metadata(component)


def dump_model(component, path):
    from ..neural_io import write_configuration

    write_configuration(component, path.parent, "tensorflow")
    component.model.save(path)


def load_model(path):
    from ..neural_io import read_configuration

    _, keras = framework()
    return read_configuration(path.parent, "tensorflow", keras.models.load_model(path, compile=False, safe_mode=True))


def restore_metadata(component, metadata):
    from ..neural_io import restore_metadata

    restore_metadata(component, metadata)
