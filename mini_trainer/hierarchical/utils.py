from collections import Counter
from collections.abc import Iterable

import torch

from mini_trainer.utils.generic import get_prior_method


def leaf_to_parents(h):
    """Construct the path from leaf-to-root for a specific leaf."""
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
    """Creates a hierarchy from the paths and class handles.

    The hierarchy is constructed based on the nodes found in the dataset.
    TODO: The hierarchy should be constructed once and saved in a structured file.

    Arguments:
        combinations: List of all leaf-to-root labels.
        class_to_idx: A mapping from classes to indexes.

    Returns:
        A list for each level of the hierarchy.
            Each list contains a list for each node containing the indices of the children of that node.
            Level 0 is the leaf level, and is not included.
    """
    n_classes = [len(class_to_idx[level]) for level in range(len(class_to_idx))]
    hierarchy = [[set() for _ in range(n)] for n in n_classes[1:]]  # Create empty lists for each level
    processed_leaves = [0] * n_classes[0]  # Keep track of which leaves have been processed

    # Iterate over the combinations
    for components in combinations:
        # Convert the class strings to indices
        indices = [class_to_idx[ctype][class_str] for ctype, class_str in enumerate(components)]

        # Skip processed leaves (species in this case)
        if processed_leaves[indices[0]] == 0:  # If the leaf has not been processed yet
            processed_leaves[indices[0]] = 1
        else:
            continue  # Skip this leaf

        # Iterate over the indices and add them to the hierarchy
        for i in range(len(indices) - 1):
            # Get the parent and child indices
            child = indices[i]
            parent = indices[i + 1]
            hierarchy[i][parent].add(child)  # Append the child to the parent's list

    return [[list(parent) for parent in level] for level in hierarchy]


def create_mask_col(indices, height, zero=-100, **kwargs):
    """Create an approximate logarithmic binary mask with the given indices.

    Arguments:
        indices (list): list of indices to include in the mask.
        height (int): Height of the mask (i.e. number of rows, also the 1+max(indices)).
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0).
            This should be a large negative number. Default: -100.
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.

    Returns:
        An approximate logarithmic binary mask for the given indices.
    """
    col = torch.zeros((height, 1), **kwargs, requires_grad=False)
    col += zero
    col[indices] = 0
    return col


def mask_islogarithmic(masks):
    """Check if a mask is contains "logarithmic" zeros and ones."""
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
    """Create approximate logarithmic binary masks for the given hierarchy.

    Arguments:
        hierarchy (list): list of lists of lists of indices.
            The first level of the list corresponds to the levels of the hierarchy,
            and each level contains a list of lists of indices for each node.
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0).
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.

    Returns:
        list of masks for each level of the hierarchy.
            Each mask has shape (n_nodes, n_child_nodes) and can be used to calculate the logits
            for the nodes based on the child logits:
            TODO: Add equation here (logarithmic matrix multiplication)
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


def batched_scatter_logsumexp(input: torch.Tensor, index: torch.Tensor, dim: int = 1):
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

    Returns:
        output: Aggregated logsumexp of ``input`` of size :math:`N x max(index)+1`.
    """
    # Scaffold tensor - same size as output
    z = input.new_zeros(shape_resize(input.shape, dim=dim, value=index.max().item() + 1))
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
