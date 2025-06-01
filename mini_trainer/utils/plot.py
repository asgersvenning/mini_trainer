import os
from random import choice
from typing import Callable, Optional, Union

import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedLocator, NullLocator
from torch import nn

from mini_trainer.classifier import last_layer_weights
from mini_trainer.utils import decimals


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
        results : dict[str, list[str]], 
        idx2cls : dict[int, str], 
        keys : tuple[str, str]=("preds", "labels"),
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
        fig, _ = plot_heatmap(conf_mat_arr, "magma", percent=False)
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
        labels : Union[list[int], torch.Tensor, np.ndarray], 
        predictions : Union[list[int], torch.Tensor, np.ndarray], 
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


def infer_heatmap_figsize(
    matrix_shape: tuple[int, int],
    min_pixels_per_cell: int = 4,
    max_total_pixels_dim: int = 8000, # Max pixels for width or height to avoid huge images
    min_figsize_dim: float = 4.0,    # Min inches for width or height
    max_figsize_dim: float = 80.0,   # Max inches for width or height
    default_dpi: float = 100.0
) -> tuple[float, float]:
    """
    Infers a suitable figsize for a heatmap based on matrix dimensions
    to ensure individual cells are not significantly blurred.

    Args:
        matrix_shape: Tuple (num_rows, num_cols) of the heatmap matrix.
        min_pixels_per_cell: Target minimum number of pixels for the shorter
                             dimension of a cell.
        max_total_pixels_dim: Maximum allowed total pixels for either width or height
                              (num_cells * min_pixels_per_cell). This helps cap
                              the calculated figure size before DPI scaling if
                              min_pixels_per_cell is very high for a large matrix.
        min_figsize_dim: Minimum size (in inches) for any dimension of the figure.
        max_figsize_dim: Maximum size (in inches) for any dimension of the figure.
        default_dpi: Assumed DPI if not available from Matplotlib's rcParams.

    Returns:
        Tuple (width_inches, height_inches) for the figure.
    """
    num_rows, num_cols = matrix_shape

    if num_rows <= 0 or num_cols <= 0:
        # Return a default small size for empty or invalid shapes
        return (min_figsize_dim, min_figsize_dim)

    try:
        # Get current DPI from Matplotlib settings if available
        dpi = plt.rcParams.get('figure.dpi', default_dpi)
    except RuntimeError: # pragma: no cover (can happen in non-GUI backends sometimes)
        dpi = default_dpi
    if dpi <= 0: dpi = default_dpi # Ensure DPI is positive


    # Calculate ideal total pixels needed for clarity
    ideal_fig_width_pixels = min(num_cols * min_pixels_per_cell, max_total_pixels_dim)
    ideal_fig_height_pixels = min(num_rows * min_pixels_per_cell, max_total_pixels_dim)

    # Convert ideal pixels to inches
    ideal_fig_width_inches = ideal_fig_width_pixels / dpi
    ideal_fig_height_inches = ideal_fig_height_pixels / dpi

    # Apply inch-based constraints
    final_fig_width = max(min_figsize_dim, min(ideal_fig_width_inches, max_figsize_dim))
    final_fig_height = max(min_figsize_dim, min(ideal_fig_height_inches, max_figsize_dim))

    return (final_fig_width, final_fig_height)

def plot_heatmap(
        mat : np.ndarray, 
        cmap_name : str='magma',
        figsize : Optional[tuple[int, int]]=None,
        font_size : int=20,
        max_ticks : int=10,
        percent : bool=True,
        min_val : Optional[int]=0,
        ax : Optional[Axes]=None,
        **kwargs
    ):
    # mask zeros so they use the 'bad' color
    masked = np.ma.masked_invalid(mat)
    if min_val is not None:
        masked = np.ma.masked_array(masked, masked <= min_val)
    
    # compute vmin from the smallest non-zero value
    masked_min = np.floor(np.log10(masked.min()))
    if np.ma.is_masked(masked_min):
        base_exp = None
    else:
        base_exp = int(masked_min)
    vmax = float(masked.max())
    if percent:
        vmax = max(1, vmax)
    
    # prepare colormap
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad('black' if cmap_name=='magma' else 'white')
    
    # plot
    new_ax = ax is None
    if new_ax:
        fig, ax = plt.subplots(figsize=figsize or infer_heatmap_figsize(mat.shape), **kwargs)
        if not isinstance(fig, Figure):
            raise RuntimeError(f"Unexpected figure type {type(fig)}")
        if not isinstance(ax, Axes):
            raise RuntimeError(f"Unexpected axes type {type(ax)}")
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    else:
        fig = plt.gcf()
    if base_exp is None:
        return fig, ax
    im = ax.imshow(masked, norm=LogNorm(vmin=10**base_exp, vmax=vmax), cmap=cmap)
    ax.axis('off')  # no axes lines, ticks, or grid
    
    # dynamic tick values: multipliers of baseline
    multipliers = [1, 1.5, 2, 3, 5, 7]
    tick_vals = [10**exp * m for exp in range(base_exp, int(np.ceil(np.log10(vmax)))) for m in multipliers]
    
    # only keep those within the data range [baseline, vmax]
    tick_vals = [v for v in tick_vals if 10**base_exp <= v <= vmax]
    
    if len(tick_vals) > max_ticks:
        tick_vals = tick_vals[::int(np.ceil(len(tick_vals)/max_ticks))]
        if percent and not bool(np.isclose(tick_vals[-1], vmax)):
            tick_vals.append(vmax)
    
    # corresponding percent labels
    labels = []
    for v in tick_vals:
        if percent:
            v *= 100
            lab = f"{int(v)}%" if float(v).is_integer() else f"{v:.2g}%"
        else:
            lab = f"{v:.{min(3, decimals(v))}f}"
        labels.append(lab)
        
    
    # place colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, ticks=tick_vals)
    cbar.ax.yaxis.set_major_locator(FixedLocator(tick_vals))
    cbar.ax.yaxis.set_minor_locator(NullLocator())
    cbar.ax.set_yticklabels(labels)
    [lbl.set_fontsize(font_size) for lbl in cbar.ax.get_yticklabels()]
    cbar.ax.tick_params(length=0)
    cbar.outline.set_visible(False)

    if new_ax:
        plt.tight_layout()
    
    return fig, ax

def class_distance(classification_weights : torch.Tensor, probability : bool=True):
    classification_weights = classification_weights.cpu().clone().detach().to(torch.float64)
    classification_weights -= classification_weights.mean(dim = 1, keepdim=True)
    classification_weights /= classification_weights.std(dim = 1, unbiased=True, keepdim=True)

    class_dmat = torch.cdist(classification_weights, classification_weights)
    if not probability:
        return class_dmat
    class_dmat_cdf = torch.distributions.Chi2(classification_weights.shape[1]).cdf((class_dmat ** 2 / 2))
    if not isinstance(class_dmat_cdf, torch.Tensor):
        raise RuntimeError(f"Unexpected CDF output type {type(class_dmat_cdf)} produced from class distance matrix.")
    return class_dmat_cdf

def plot_model_class_distance(model : nn.Module, **kwargs):
    llw = last_layer_weights(model)
    cdm = class_distance(llw, True)
    cdm.fill_diagonal_(torch.nan)
    return plot_heatmap(cdm, **kwargs)