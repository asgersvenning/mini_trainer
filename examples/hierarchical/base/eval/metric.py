import json
from typing import Union

import numpy as np
from hierarchical.base.gbif import resolve_id
from hierarchical.base.integration import DEFAULT_HIERARCHY_LEVELS
from tqdm import tqdm as TQDM

from itertools import chain

from mini_trainer.utils.plot import raw_confusion_matrix, plot_heatmap


def rank_error(
        predictions : Union[list[Union[str, int]], list[tuple[Union[str, int], ...]]], 
        labels : list[Union[int, str]],
        progress : bool=False
    ):
    ranks = []
    elements = zip(predictions, labels)
    if progress:
        elements = TQDM(elements, total=len(labels))
    for ps, ls in elements:
        if not isinstance(ps, (list, tuple, dict)):
            ps = resolve_id(ps).values()
        ls = resolve_id(ls).values()
        for lvl, (p, l) in enumerate(zip(ps, ls)):
            if p == l:
                break
        ranks.append(lvl)
    # return Counter(ranks)
    return sum(ranks) / len(ranks)


def confusion_matrices(
        predictions : Union[list[Union[str, int]], list[tuple[Union[str, int], ...]]], 
        labels : list[Union[int, str]], 
        levels : int=len(DEFAULT_HIERARCHY_LEVELS),
        progress : bool=False
    ):
    cf_mats = []
    pred_long, lab_long = [[[] for _ in range(levels)] for _ in range(2)]
    elements = zip(predictions, labels)
    if progress:
        elements = TQDM(elements, total=len(labels))
    for ps, ls in elements:
        if not isinstance(ps, (list, tuple, dict)):
            ps = resolve_id(ps).values()
        ls = resolve_id(ls).values()
        for lvl, (p, l) in enumerate(zip(ps, ls)):
            pred_long[lvl].append(p)
            lab_long[lvl].append(l)
    comb = sorted(set(list(chain(zip(*pred_long), zip(*lab_long)))))
    return comb