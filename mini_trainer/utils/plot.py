import os
from random import choice
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
from torch import nn

from mini_trainer.classifier import last_layer_weights


def debug_augmentation(
        augmentation : Callable[[torch.Tensor], torch.Tensor],
        dataset : torch.utils.data.Dataset,
        output_dir : Optional[str]=None,
        strict : bool=True
    ):
    try:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))

        example_image : torch.Tensor = dataset[choice(range(len(dataset)))][0].clone().float().cpu()
        
        axs[0].imshow(example_image.permute(1,2,0))
        axs[1].imshow(augmentation(example_image).permute(1,2,0))

        plt.savefig(os.path.join(output_dir, "example_augmentation.png") if output_dir is not None else "example_augmentation.png")
        plt.close()
    except Exception as e:
        e_msg = (
            "Error while attempting to create debug augmentation image."
            "Perhaps the supplied dataloader doesn't return items (image, label) in the expected format."
        )
        e.add_note(e_msg)
        if strict:
            raise e
        print(e_msg)
        return False
    return True


def named_confusion_matrix(
        results : Dict[str, List[str]], 
        idx2cls : Dict[int, str], 
        keys : Tuple[str, str]=("preds", "labels"),
        verbose : bool=False, 
        plot_conf_mat : Union[bool, str]=False
    ):
    # Build confusion matrix and compute accuracies
    classes = [idx2cls[i] for i in sorted(list(idx2cls))]

    # Initialize confusion matrix and counters
    conf_mat = {lab: {pred: 0 for pred in classes} for lab in classes}
    total_correct = 0
    total_samples = 0
    per_class_total = {cls: 0 for cls in classes}
    per_class_correct = {cls: 0 for cls in classes}

    # Populate confusion matrix and count correct predictions
    for p, l in zip(results[keys[0]], results[keys[1]]):
        conf_mat[l][p] += 1
        total_samples += 1
        per_class_total[l] += 1
        if l.lower().strip() == p.lower().strip():
            total_correct += 1
            per_class_correct[l] += 1

    if plot_conf_mat:
        conf_mat_arr = np.array([[conf_mat[g][p] for p in classes] for g in classes]).astype(np.float64)
        fig, _ = plot_heatmap(conf_mat_arr, "magma")
        if isinstance(plot_conf_mat, bool):
            plot_conf_mat = "confusion_matrix.png"
        fig.savefig(plot_conf_mat)
        plt.close(fig)

    # Print the confusion matrix (numbers only, aligned)
    if verbose:
        max_cf_n = max(val for d in conf_mat.values() for val in d.values())
        width = len(str(max_cf_n))
        for lab in classes:
            row_str = "|".join(
                "{:>{width}d}".format(conf_mat[lab][pred], width=width)
                if conf_mat[lab][pred] != 0
                else " " * width
                for pred in classes
            )
            print(row_str)

    # Compute and print per-class accuracies
    if verbose:
        print("\nPer-class Accuracies:")
    macro_acc = 0.0
    num_class_with_any = 0
    for cls in classes:
        if per_class_total[cls] > 0:
            acc = per_class_correct[cls] / per_class_total[cls]
            num_class_with_any += 1
        else:
            acc = 0.0
        macro_acc += acc
        if verbose:
            print(f"{cls:_<{max(map(len, classes))}}{acc:_>9.1%} ({per_class_correct[cls]}/{per_class_total[cls]})")
    macro_acc /= num_class_with_any or 1

    # Micro accuracy: overall correct predictions / total predictions
    micro_acc = total_correct / total_samples if total_samples > 0 else 0.0

    if verbose:
        print(f"\nMicro Accuracy: {micro_acc:.2%} ({total_correct}/{total_samples})")
        print(f"Macro Accuracy: {macro_acc:.2%}")
    
    return {
        "hits" : per_class_correct,
        "totals" : per_class_total,
        "micro" : micro_acc,
        "macro" : macro_acc,
        "conf_mat" : conf_mat
    }

def raw_confusion_matrix(
        labels : Union[List[int], torch.Tensor, np.ndarray], 
        predictions : Union[List[int], torch.Tensor, np.ndarray], 
        n_classes : int
    ):
    labels = np.asarray(labels).ravel().astype(np.int64)
    predictions = np.asarray(predictions).ravel().astype(np.int64)

    indices = n_classes * labels + predictions
    cm = np.bincount(indices, minlength=n_classes * n_classes)
    cm = cm.reshape((n_classes, n_classes)).astype(np.float64)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / row_sums
        cm_norm[~np.isfinite(cm_norm)] = 0.0  # set inf and NaN to zero

    return cm_norm

def plot_heatmap(
        mat : np.ndarray, 
        cmap_name : str='magma',
        figsize : Tuple[int, int]=(20, 20),
        font_size : int=20,
        max_ticks : int=10,
        **kwargs
    ):
    # mask zeros so they use the 'bad' color
    masked = np.ma.masked_invalid(np.ma.masked_equal(mat, 0))
    
    # compute vmin from the smallest non-zero value
    base_exp = int(np.floor(np.log10(masked.min())))
    vmax = float(max(1, masked.max()))
    
    # prepare colormap
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('black' if cmap_name=='magma' else 'white')
    
    # plot
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    if not isinstance(fig, Figure):
        raise RuntimeError(f"Unexpected figure type {type(fig)}")
    if not isinstance(ax, Axes):
        raise RuntimeError(f"Unexpected axes type {type(ax)}")
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    im = ax.imshow(masked, norm=LogNorm(vmin=10**base_exp, vmax=vmax), cmap=cmap)
    ax.axis('off')  # no axes lines, ticks, or grid
    
    # dynamic tick values: multipliers of baseline
    multipliers = [1, 1.5, 2, 3, 5, 7]
    tick_vals = [10**exp * m for exp in range(base_exp, 1) for m in multipliers]
    if len(tick_vals) > max_ticks:
        tick_vals = tick_vals[::int(np.ceil(len(tick_vals)/max_ticks))]
        if not bool(np.isclose(tick_vals[-1], 1)):
            tick_vals.append(1)
    # only keep those within the data range [baseline, vmax]
    tick_vals = [v for v in tick_vals if 10**base_exp <= v <= vmax]
    
    # corresponding percent labels
    labels = []
    for v in tick_vals:
        v *= 100
        lab = f"{int(v)}%" if float(v).is_integer() else f"{v:.2g}%"
        labels.append(lab)
    
    # place colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=tick_vals)
    cbar.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.set_yticklabels(labels)
    [lbl.set_fontsize(font_size) for lbl in cbar.ax.get_yticklabels()]
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)

    plt.tight_layout()
    
    return fig, ax

def class_distance(classification_weights : torch.Tensor, probability : bool=True):
    classification_weights = classification_weights.cpu().clone().to(torch.float64)
    classification_weights -= classification_weights.mean(dim = 1, keepdim=True)
    classification_weights /= classification_weights.std(dim = 1, unbiased=True, keepdim=True)

    class_dmat = torch.cdist(classification_weights, classification_weights)
    if not probability:
        return class_dmat
    class_dmat_cdf = torch.distributions.Chi2(classification_weights.shape[1]).cdf((class_dmat ** 2 / 2))
    if not isinstance(class_dmat_cdf, torch.Tensor):
        raise RuntimeError(f"Unexpected CDF output type {type(class_dmat_cdf)} produced from class distance matrix.")
    return class_dmat_cdf

def plot_model_class_distance(model : nn.Module):
    llw = last_layer_weights(model)
    cdm = class_distance(llw, True)
    cdm.fill_diagonal_(torch.nan)
    return plot_heatmap(cdm, figsize=(5, 5), font_size=5)