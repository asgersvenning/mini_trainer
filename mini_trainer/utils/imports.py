import importlib


def class_path(x) -> str:
    cls = x if isinstance(x, type) else type(x)
    return f"{cls.__module__}:{cls.__qualname__}"


def import_class(path: str) -> type:
    module_name, qualname = path.split(":", 1)
    obj = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj # type: ignore
