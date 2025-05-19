from typing import Iterable

import torch

def leaf_to_parents(h):
    l2p = []
    p2c = None
    for lvl in h:
        c2p = {e : i for i, p in enumerate(lvl) for e in p}
        if p2c is not None:
            c2p = {c : v for k, v in c2p.items() for c in p2c[k]}
        p2c = dict()
        for p, c in c2p.items():
            if c not in p2c:
                p2c[c] = []
            p2c[c].append(p)
        l2p.append({k : v for k, v in sorted(c2p.items())})
    return l2p

def create_hierarchy(combinations : Iterable[list[str]], class_to_idx : list[dict[str, int]]) -> list[list[list[int]]]:
    """
    Creates a hierarchy from the paths and class handles.
    The hierarchy is constructed based on the nodes found in the dataset. 
    TODO: The hierarchy should be constructed once and saved in a structured file.

    Arguments:
        TODO

    Returns:
        list: A list for each level of the hierarchy. Each list contains a list for each node containing the indices of the children of that node. 
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
            continue # Skip this leaf
        
        # Iterate over the indices and add them to the hierarchy
        for i in range(len(indices) - 1):
            # Get the parent and child indices
            child = indices[i]
            parent = indices[i+1]
            hierarchy[i][parent].add(child)  # Append the child to the parent's list

    return [[list(parent) for parent in level] for level in hierarchy]

def create_mask_col(indices, height, zero=-100, **kwargs):
    """
    Create an approximate logarithmic binary mask with the given indices.

    Arguments:
        indices (list): list of indices to include in the mask.
        height (int): Height of the mask (i.e. number of rows, also the 1+max(indices)).
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0). 
            This should be a large negative number. Default: -100.
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.
    
    Returns:
        torch.Tensor: An approximate logarithmic binary mask for the given indices.
    """
    col = torch.zeros((height, 1), **kwargs, requires_grad=False)
    col += zero
    col[indices] = 0
    return col

def mask_islogarithmic(masks):
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
    """
    Create approximate logarithmic binary masks for the given hierarchy.

    Arguments:
        hierarchy (list): list of lists of lists of indices. 
            The first level of the list corresponds to the levels of the hierarchy, and each level contains a list of lists of indices for each node.
        zero (int): "Approximate zero" value. This is used to avoid numerical issues with log(0).
        **kwargs: Keyword arguments to pass to torch.zeros(). Notably 'device' and 'dtype'.

    Returns:
        list: list of masks for each level of the hierarchy.
            Each mask has shape (n_nodes, n_child_nodes) and can be used to calculate the logits for the nodes based on the child logits:
            TODO: Add equation here (logarithmic matrix multiplication)
    """
    masks = []
    for level in hierarchy:
        n = sum([len(indices) for indices in level])
        masks.append([create_mask_col(indices, n, zero=zero, **kwargs) for indices in level])

    return [torch.hstack(level) for level in masks]