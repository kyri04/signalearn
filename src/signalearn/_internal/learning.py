from dask import delayed


def validate_fit_parameters(parameters):
    if "y" in parameters:
        raise TypeError("Supply supervised targets through target as a SignaLearn Field")


def deferred_prediction(adapter, model, inputs, length, schema, output=None):
    component = adapter.snapshot(model)
    return delayed(adapter.apply_prediction)(component, inputs[0], inputs[2], length, schema, output)
