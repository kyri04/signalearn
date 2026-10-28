def get_adapter(model, operation):
    from . import component, sklearn

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
    from . import sklearn

    if name == sklearn.persistence_name:
        return sklearn
    raise ValueError(f"No persistence integration supports adapter {name!r}")
