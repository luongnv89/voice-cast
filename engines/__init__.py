from importlib import import_module

_engine_classes: dict[str, type] = {}


def __getattr__(name: str):
    if name in _engine_classes:
        return _engine_classes[name]
    module_map = {
        "CoquiEngine": ".coqui_engine",
        "ChatterboxEngine": ".chatterbox_engine",
        "Audio8Engine": ".audio8_engine",
    }
    if name in module_map:
        mod = import_module(module_map[name], __package__)
        cls = getattr(mod, name)
        _engine_classes[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["CoquiEngine", "ChatterboxEngine", "Audio8Engine"]
