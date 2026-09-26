from collections import Counter
from collections.abc import Iterable

import torch

from mini_trainer.modeling import get_prior_method


def leaf_to_parents(h):
    """Return a leaf-to-ancestor index mapping for each parent rank."""
    l2p = []
    p2c = None
    for lvl in h:
        c2p = {e: i for i, p in enumerate(lvl) for e in p}
        if p2c is not None:
            c2p = {c: v for k, v in c2p.items() for c in p2c[k]}
        p2c = dict()
        for p, c in c2p.items():
            if c not in p2c:
                p2c[c] = []
            p2c[c].append(p)
        l2p.append({k: v for k, v in sorted(c2p.items())})
    return l2p


def create_hierarchy(combinations: Iterable[list[str]], class_to_idx: list[dict[str, int]]) -> list[list[list[int]]]:
    """Build parent-to-child index lists, ordered from the first parent rank upward.

    Paths and class mappings are leaf-first. Only the first path for each leaf
    contributes; repeated leaves are ignored.
    """
    n_classes = [len(class_to_idx[level]) for level in range(len(class_to_idx))]
    hierarchy = [[set() for _ in range(n)] for n in n_classes[1:]]
    processed_leaves = [0] * n_classes[0]

    for components in combinations:
        indices = [class_to_idx[ctype][class_str] for ctype, class_str in enumerate(components)]

        if processed_leaves[indices[0]] == 0:
            processed_leaves[indices[0]] = 1
        else:
            continue

        for i in range(len(indices) - 1):
            child = indices[i]
            parent = indices[i + 1]
            hierarchy[i][parent].add(child)

    return [[list(parent) for parent in level] for level in hierarchy]


def create_mask_col(indices, height, zero=-100, **kwargs):
    """Return a (height, 1) additive mask: 0 at indices, ``zero`` elsewhere.

    ``zero`` defaults to -100 as a finite approximation to log(0). Keyword arguments
    such as device and dtype are forwarded to torch.zeros.
    """
    col = torch.zeros((height, 1), **kwargs, requires_grad=False)
    col += zero
    col[indices] = 0
    return col


def mask_islogarithmic(masks):
    """Detect values outside {0, 1}; reject lists mixing binary and additive masks."""
    if isinstance(masks, list):
        response = [mask_islogarithmic(mask) for mask in masks]
        all_true = all(response)
        all_false = not any(response)
        ambiguous = not all_true and not all_false
        if ambiguous:
            raise ValueError("Some masks are logarithmic and some are not.")
        return all_true
    return not torch.all((masks == 0) | (masks == 1))


def mask_hierarchy(hierarchy, zero=-100, **kwargs):
    """Return one additive (children, parents) mask per hierarchy rank.

    Entries are 0 for child-parent membership and ``zero`` otherwise (default -100).
    Children must have contiguous indices and belong to exactly one parent.
    Device and dtype keyword arguments are forwarded to torch.zeros.
    """
    masks = []
    for level in hierarchy:
        n = sum([len(indices) for indices in level])
        masks.append([create_mask_col(indices, n, zero=zero, **kwargs) for indices in level])

    return [torch.hstack(level) for level in masks]


def shape_resize(shape: torch.Size | list[int], dim: int, value: int):  # noqa: D103
    shape = list(shape)
    shape[dim] = value
    return shape


def batched_scatter_logsumexp(input: torch.Tensor, index: torch.Tensor, dim: int = 1, dim_size: int | None = None):
    """Aggregates the elements of the ``input`` tensor with an index along a dimension using logsumexp.

    ```
    out[j][i] = input[j][index == i].logsumexp()
    ```

    OBS: Behavior for indexes that do not contain all integers from
        0 to :math:`max(index)` or when ``dim`` is not 1 is not defined.

    Args:
        input: Input tensor of size :math:`N x K`.
        index: Long-Tensor of size :math:`K` containing the elements along ``dim`` in ``input`` to aggregate.
        dim: Dimension to aggregate over (default=1).
        dim_size: Custom output dimension size along dim (default=None, dynamically computed).

    Returns:
        output: Aggregated logsumexp of ``input`` of size :math:`N x max(index)+1`.
    """
    if dim_size is None:
        dim_size = int(index.max().item() + 1)
    z = torch.zeros(shape_resize(input.shape, dim=dim, value=dim_size), dtype=input.dtype, device=input.device)
    index = index.expand_as(input)
    c = z.scatter_reduce(dim=dim, index=index, src=input, reduce="amax", include_self=False)
    return z.scatter_add(dim=dim, index=index, src=(input - c.gather(dim=dim, index=index)).exp()).log() + c


def prior_from_labels(labels: list[list[int]] | list[int], cls2idx: dict, method: str = "adjust", **kwargs):
    if isinstance(labels[0], int):
        raise ValueError("Expected hierarchical labels, but got flat.")
    ncls = [len(cls2idx[str(lvl)]) for lvl in range(len(cls2idx))]
    nlvls = len(ncls)
    counts = {lvl: Counter([lab[lvl] for lab in labels]) for lvl in range(nlvls)}
    counts = {k: [v.get(i, 0) for i in range(ncls[int(k)])] for k, v in counts.items()}
    func = get_prior_method(method)
    return [func(counts[lvl], **kwargs) for lvl in range(nlvls)]
