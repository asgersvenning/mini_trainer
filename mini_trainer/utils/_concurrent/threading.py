import inspect
import types
from collections.abc import Callable, Iterable
from functools import wraps
from typing import Annotated, Any, Concatenate, ParamSpec, TypeVar, cast, get_args, get_origin, get_type_hints, overload

from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map as _thread_map

from mini_trainer import get_logger

X = TypeVar("X")  # input value
R = TypeVar("R")  # return value
P = ParamSpec("P")


def first_arg_base_types(fn):  # noqa: D103
    params = tuple(inspect.signature(fn).parameters.values())
    if not params:
        return [Any]
    first = params[0].name
    hints = get_type_hints(fn)
    ann = hints.get(first, Any)

    def strip_annotated(t):
        while get_origin(t) is Annotated:
            t = get_args(t)[0]
        return t

    def base(t):
        t = strip_annotated(t)
        o = get_origin(t)
        return o or t

    def flatten_union(t):
        t = strip_annotated(t)
        o = get_origin(t)
        if o in (types.UnionType, getattr(types, "NoneType", type(None)), None):
            pass
        if o is types.UnionType or o is getattr(__import__("typing"), "Union"):
            return [base(x) for x in get_args(t)]
        return [base(t)]

    out = flatten_union(ann)
    return list(dict.fromkeys(out))


def thread_map[X, R](func: Callable[[X], R], it: Iterable[X], **kwargs: Any) -> list[R]:
    """Type helper that fixes return type annotation from ``tqdm.contrib.concurrent.thread_map``."""
    return cast(list[R], _thread_map(func, it, **kwargs))


def multithread_vectorize(leave: bool = False, min_items_to_multithread: int = 32, **tqdm_kwargs: Any):
    """Decorator to vectorize a function on it's first argument."""

    def decorator[X, R](f: Callable[Concatenate[X, P], R]):
        _base_types = first_arg_base_types(f)
        collected_types: list[type[Any]] = []
        for t in _base_types:
            if not isinstance(t, type):
                raise TypeError(
                    "`multithread_vectorize` decorator only supports simple argument type annotation "
                    f"for the vectorized arguments, not: `{t}`"
                )
            if t in (list, tuple):
                get_logger().warning(
                    "Using `list` or `tuple` as the base-type with `multithread_vectorize` is dangerous and might not work as you expect."
                )
            collected_types.append(t)
        base_types = tuple(collected_types)

        @overload
        def wrapped(x: X, /, *args: P.args, **kwargs: P.kwargs) -> R: ...
        @overload
        def wrapped(x: Iterable[X], /, *args: P.args, **kwargs: P.kwargs) -> list[R]: ...
        @wraps(f)
        def wrapped(x: X | Iterable[X], /, *args: P.args, **kwargs: P.kwargs) -> R | list[R]:
            if isinstance(x, base_types):
                return f(x, *args, **kwargs)
            if callable(getattr(x, "__len__", None)) and len(x) < min_items_to_multithread:
                return [f(y, *args, **kwargs) for y in tqdm(x, leave=leave, **tqdm_kwargs)]
            return cast(list[R], _thread_map(lambda y: f(y, *args, **kwargs), x, leave=leave, **tqdm_kwargs))

        return wrapped

    return decorator
