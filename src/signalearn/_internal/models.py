import sys


def native_framework(model):
    keras = sys.modules.get("keras")
    if keras is not None and isinstance(model, keras.Model):
        return "tensorflow"
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(model, torch.nn.Module):
        return "torch"
    raise TypeError("Model requires a native Keras Model or torch.nn.Module instance")


def configure_model(component):
    framework = native_framework(component.model)
    if framework == "tensorflow":
        for name in ("loss", "optimizer", "factory", "factory_params"):
            if getattr(component, name) is not None:
                raise TypeError(f"{name} is a PyTorch-only option; configure native Keras training with compile()")
    if component.device is None:
        component.device = "cpu" if framework == "torch" else "/CPU:0"
    return framework
