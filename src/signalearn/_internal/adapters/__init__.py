def get_adapter(model, operation):
    from ...components import Model
    from . import component, sklearn

    if isinstance(model, Model):
        from ..models import configure_model
        from . import tensorflow, torch

        return torch if configure_model(model) == "torch" else tensorflow

    if component.supports(model):
        if operation == "save_model":
            raise TypeError("Persistence is not supported for custom transform components; "
                            "provide a persistence integration for this component")
        if operation != "transform":
            raise TypeError("The custom component integration currently supports transform only; "
                            "fit the component externally if it learns parameters")
        return component
    if sklearn.supports(model, operation):
        return sklearn
    raise TypeError(f"No integration supports {operation} for {type(model).__name__}; "
                    "use the scikit-learn estimator interface or the documented custom transform contract")


def get_persistence_adapter(name):
    from . import sklearn, tensorflow, torch

    if name == sklearn.persistence_name:
        return sklearn
    if name in {torch.persistence_name, tensorflow.persistence_name}:
        return torch if name == torch.persistence_name else tensorflow
    raise ValueError(f"No persistence integration supports adapter {name!r}")
