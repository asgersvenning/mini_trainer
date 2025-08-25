from itertools import chain

from tqdm import tqdm as TQDM

from mini_trainer.hierarchical.gbif import resolve_id
from mini_trainer.hierarchical.integration import DEFAULT_HIERARCHY_LEVELS


def rank_error(
        predictions : list[str | int] | list[tuple[str | int, ...]], 
        labels : list[int | str],
        progress : bool=False
    ):
    ranks = []
    elements = zip(predictions, labels)
    if progress:
        elements = TQDM(elements, total=len(labels))
    for predictions, labels in elements:
        if not isinstance(predictions, (list, tuple, dict)):
            predictions = resolve_id(predictions).values()
        labels = resolve_id(labels).values()
        for level, (prediction, label) in enumerate(zip(predictions, labels)):
            if prediction == label:
                break
        ranks.append(level)
    # return Counter(ranks)
    return sum(ranks) / len(ranks)


def confusion_matrices(
        predictions : list[str | int] | list[tuple[str | int, ...]], 
        labels : list[int | str], 
        levels : int=len(DEFAULT_HIERARCHY_LEVELS),
        progress : bool=False
    ):
    cf_mats = []
    pred_long, lab_long = [[[] for _ in range(levels)] for _ in range(2)]
    elements = zip(predictions, labels)
    if progress:
        elements = TQDM(elements, total=len(labels))
    for predictions, labels in elements:
        if not isinstance(predictions, (list, tuple, dict)):
            predictions = resolve_id(predictions).values()
        labels = resolve_id(labels).values()
        for lvl, (prediction, label) in enumerate(zip(predictions, labels)):
            pred_long[lvl].append(prediction)
            lab_long[lvl].append(label)
    comb = sorted(set(list(chain(zip(*pred_long), zip(*lab_long)))))
    return comb