import math
import os
from random import choice
from typing import Callable, Optional, Union

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.backends import backend_agg
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullLocator
from torch import nn
from torchvision.transforms.functional import resize

from mini_trainer.classifier import last_layer_weights
from mini_trainer.utils import decimals
from PIL.Image import fromarray


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
        arr = plot_heatmap(conf_mat_arr, "magma", percent=False)
        if isinstance(plot_conf_mat, bool):
            plot_conf_mat = "confusion_matrix.png"
        fromarray(arr).save(plot_conf_mat)

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
        n_classes : Optional[int]=None
    ):
    labels = np.asarray(labels).ravel().astype(np.int64)
    predictions = np.asarray(predictions).ravel().astype(np.int64)
    if n_classes is None:
        n_classes = int(max(max(labels), max(predictions))) + 1

    indices = n_classes * labels + predictions
    cm = np.bincount(indices, minlength=n_classes * n_classes)
    cm = cm.reshape((n_classes, n_classes)).astype(np.float64)

    row_sums = cm.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / row_sums
        cm_norm[~np.isfinite(cm_norm)] = 0.0  # set inf and NaN to zero

    return cm_norm


# --- Constants ---
MIN_DISPLAY_DIM_HEATMAP = 500
MAX_DISPLAY_DIM_HEATMAP = 5000
COLORBAR_RENDER_DPI = 150
COLORBAR_TARGET_WIDTH_PIXELS = 200 # Approximate width for the colorbar image

# --- Helper: Matrix Aggregation ---
def _aggregate_matrix_max(matrix: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    """Aggregates matrix by summing values in blocks. Handles non-divisible shapes by padding."""
    orig_rows, orig_cols = matrix.shape
    block_rows, block_cols = block_shape
    
    pad_rows = (block_rows - orig_rows % block_rows) % block_rows
    pad_cols = (block_cols - orig_cols % block_cols) % block_cols
    
    padded_matrix = matrix
    if pad_rows > 0 or pad_cols > 0:
        padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    
    new_rows, new_cols = padded_matrix.shape
    target_rows, target_cols = new_rows // block_rows, new_cols // block_cols
    
    # Efficient reshape and sum for block aggregation
    return padded_matrix.reshape(target_rows, block_rows, target_cols, block_cols).max(axis=3).max(axis=1)

# --- Helper: Matrix Scaling ---
def _get_scaled_matrix_for_display(mat: np.ndarray) -> np.ndarray:
    """Resizes matrix: downscales then upscales to fit display dimension constraints."""
    processed_mat = mat 
    orig_rows, orig_cols = mat.shape

    # 1. Downscale to fit MAX_DISPLAY_DIM_HEATMAP
    block_r = math.ceil(orig_rows / MAX_DISPLAY_DIM_HEATMAP) if orig_rows > MAX_DISPLAY_DIM_HEATMAP else 1
    block_c = math.ceil(orig_cols / MAX_DISPLAY_DIM_HEATMAP) if orig_cols > MAX_DISPLAY_DIM_HEATMAP else 1

    if block_r > 1 or block_c > 1:
        if processed_mat is mat: processed_mat = mat.copy() # Copy if modifying
        processed_mat = _aggregate_matrix_max(processed_mat, (block_r, block_c))
    
    curr_rows, curr_cols = processed_mat.shape

    # 2. Upscale to meet MIN_DISPLAY_DIM_HEATMAP, constrained by MAX_DISPLAY_DIM_HEATMAP
    k_ideal = max(
        math.ceil(MIN_DISPLAY_DIM_HEATMAP / curr_rows) if curr_rows > 0 and curr_rows < MIN_DISPLAY_DIM_HEATMAP else 1,
        math.ceil(MIN_DISPLAY_DIM_HEATMAP / curr_cols) if curr_cols > 0 and curr_cols < MIN_DISPLAY_DIM_HEATMAP else 1
    )
    k_cap = min(
        math.floor(MAX_DISPLAY_DIM_HEATMAP / curr_rows) if curr_rows > 0 else float('inf'),
        math.floor(MAX_DISPLAY_DIM_HEATMAP / curr_cols) if curr_cols > 0 else float('inf')
    )
    final_k = int(max(1, min(k_ideal, k_cap)))

    if final_k > 1:
        if processed_mat is mat: processed_mat = mat.copy() # Copy if modifying
        return np.kron(processed_mat, np.ones((final_k, final_k), dtype=processed_mat.dtype))
    
    return processed_mat.copy() if processed_mat is mat else processed_mat


# --- Helper: Heatmap Array Generation ---
def _generate_heatmap_rgb_array(
    display_mat: np.ndarray, 
    min_val_display: Optional[float],
    cmap_name: str,
    percent: bool
) -> tuple[Optional[np.ndarray], Optional[LogNorm], float, float]:
    """Generates the RGB heatmap image array using Matplotlib colormaps, and returns norm info."""
    
    masked_data = np.ma.masked_invalid(display_mat.astype(float)) # Handle NaNs
    if min_val_display is not None:
        masked_data = np.ma.masked_less_equal(masked_data, min_val_display)

    positive_values = masked_data.compressed()
    positive_values = positive_values[positive_values > 0]

    if positive_values.size == 0:
        # Create a dummy black image if no valid data
        h, w = display_mat.shape
        dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return dummy_rgb, None, 0., 1.

    min_positive_val = positive_values.min()
    norm_vmin = 10**math.floor(math.log10(min_positive_val))
    norm_vmax = float(masked_data.max()) # Max of the *original* masked_data, not just positive

    if percent: # Adjust norm for percentages
        norm_vmax = max(1.0, norm_vmax)
    
    # Handle edge case where vmin might equal or exceed vmax after adjustments
    if norm_vmin >= norm_vmax:
        if norm_vmin > 0: 
            norm_vmax = norm_vmin * (1.1 if norm_vmin > 1 else 2.0) # Create small range
        else: # Cannot make a valid LogNorm
            h, w = display_mat.shape
            dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
            return dummy_rgb, None, 0., 1.
    
    norm = LogNorm(vmin=norm_vmin, vmax=norm_vmax)
    cmap = mpl.colormaps[cmap_name].copy() # Get a mutable copy
    cmap.set_bad(color=(0,0,0) if cmap_name == 'magma' else (1,1,1), alpha=1.0)
    
    try:
        normalized_values = norm(masked_data)
    except ValueError as e: # Can happen if norm range is invalid
        print(f"Warning: LogNorm failed ({e}). Returning empty image.")
        # Fallback or re-raise, for now, dummy image
        h, w = display_mat.shape
        dummy_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        return dummy_rgb, None, 0., 1.

    rgba_image_float = cmap(normalized_values) # (H, W, 4) float RGBA
    
    # Convert to RGB uint8, dropping alpha
    rgb_image_uint8 = (rgba_image_float[:, :, :3] * 255).astype(np.uint8)
    
    return rgb_image_uint8, norm, norm_vmin, norm_vmax

# --- Helper: Colorbar Ticks ---
def _get_colorbar_ticks_and_labels(
    norm_vmin: float, norm_vmax: float, max_ticks: int, percent: bool
) -> tuple[list[float], list[str]]:
    """Generates tick values and labels for the colorbar."""
    if not (norm_vmin > 0 and norm_vmax > 0 and norm_vmin < norm_vmax): return [], []

    num_decades = math.log10(norm_vmax / norm_vmin) if norm_vmin > 0 and norm_vmax > 0 else 1
    multipliers = [1, 2, 5] if num_decades < 2 else [1, 1.5, 2, 3, 5, 7] #Fewer for small ranges
    
    tick_cands = {norm_vmin, norm_vmax}
    start_exp = math.floor(math.log10(norm_vmin)) if norm_vmin > 0 else 0
    end_exp = math.ceil(math.log10(norm_vmax)) if norm_vmax > 0 else 0

    for exp_val in range(start_exp, end_exp + 1):
        for m in multipliers:
            tick = (10**exp_val) * m
            if norm_vmin <= tick <= norm_vmax: # Ensure ticks are within actual data range
                tick_cands.add(tick)
    
    # Filter again to be absolutely sure, then sort
    sorted_ticks = sorted(list(t for t in tick_cands if norm_vmin <= t <= norm_vmax))

    if len(sorted_ticks) > max_ticks: # Subsample if too many
        indices = np.round(np.linspace(0, len(sorted_ticks) - 1, max_ticks)).astype(int)
        final_ticks = [sorted_ticks[i] for i in sorted(list(set(indices)))]
        # Ensure original vmin and vmax are considered if space allows
        if max_ticks >= 1 and not np.isclose(final_ticks[0], norm_vmin): final_ticks.insert(0,norm_vmin)
        if max_ticks >= 2 and not np.isclose(final_ticks[-1], norm_vmax): final_ticks.append(norm_vmax)
        final_ticks = sorted(list(set(t for t in final_ticks if norm_vmin <= t <= norm_vmax)))[:max_ticks]
    else:
        final_ticks = sorted_ticks
    
    # Ensure at least two ticks (min/max) if possible, if list became empty by max_ticks=0 or 1
    if not final_ticks and len(sorted_ticks) >= 1:
        final_ticks = [sorted_ticks[0]]
        if len(sorted_ticks) > 1: final_ticks.append(sorted_ticks[-1])
        final_ticks = sorted(list(set(final_ticks)))


    labels = []
    for v_tick in final_ticks:
        val_fmt = v_tick * 100 if percent else v_tick
        lab_str = ""
        if percent:
            if abs(val_fmt) < 0.01 and val_fmt != 0: lab_str = f"{val_fmt:.1e}%"
            elif np.isclose(val_fmt, round(val_fmt)): lab_str = f"{int(round(val_fmt))}%"
            else: lab_str = f"{val_fmt:.2g}%"
        else: # Non-percent
            if abs(v_tick) >= 1000 or (abs(v_tick) < 0.001 and v_tick != 0): lab_str = f"{v_tick:.2g}"
            else: lab_str = f"{v_tick:.3g}".rstrip('0').rstrip('.') # General, remove trailing .0
        labels.append(lab_str)
        
    return final_ticks, labels

# --- Helper: Colorbar Array Generation ---
def _generate_colorbar_rgb_array(
    norm_obj: mpl.colors.LogNorm, 
    cmap_name_str: str, 
    tick_list: list[float], 
    tick_label_list: list[str],
    target_height_pixels: int, 
    font_size_pt: int
) -> np.ndarray:
    """Renders a colorbar using Matplotlib to an RGB NumPy array of target_height_pixels."""
    fig_width_inches = COLORBAR_TARGET_WIDTH_PIXELS / COLORBAR_RENDER_DPI
    fig_height_inches = target_height_pixels / COLORBAR_RENDER_DPI

    fig_cbar = mpl.figure.Figure(figsize=(fig_width_inches, fig_height_inches), dpi=COLORBAR_RENDER_DPI)
    canvas_cbar = backend_agg.FigureCanvasAgg(fig_cbar)
    
    ax_cbar_rect = [0.15, 0.05, 0.3, 0.9] # [left, bottom, width_of_strip, height_of_strip]
    ax_cbar = fig_cbar.add_axes(ax_cbar_rect)
    cmap_obj_for_cbar = mpl.colormaps[cmap_name_str] # Fresh colormap for cbar
    
    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap_obj_for_cbar, norm=norm_obj, 
                                   orientation='vertical', ticks=tick_list)
    
    if tick_list: # Only set labels if there are ticks
        cb.set_ticklabels(tick_label_list)
        cb.ax.tick_params(labelsize=font_size_pt, length=0) # No tick lines
        cb.outline.set_visible(False)
    else: # No ticks, make it a plain strip
        ax_cbar.set_axis_off()

    canvas_cbar.draw() # Render the figure
    img_rgb = np.asarray(canvas_cbar.buffer_rgba())[:, :, :3] # Get buffer and convert to ndarray

    if img_rgb.shape[0] != target_height_pixels:
        img_rgb = resize(torch.tensor(img_rgb).permute(2, 0, 1), (target_height_pixels, round(target_height_pixels / img_rgb.shape[0] * img_rgb.shape[1]))).permute(1, 2, 0).numpy()

    plt.close(fig_cbar)
    return img_rgb


# --- Main Plotting Function ---
def plot_heatmap(
    mat : Union[np.ndarray, torch.Tensor],
    cmap_name : str='magma',
    font_size : int=10, # For colorbar labels
    max_colorbar_ticks : int=8,
    percent : bool=True,
    min_val_display : Optional[float]=None,
    colorbar : bool=True
) -> Optional[np.ndarray]:
    """
    Plots a high-resolution confusion matrix using NumPy and Matplotlib.
    Returns a combined RGB NumPy array (heatmap + colorbar), or None for empty input.
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().detach().numpy()
    
    if mat.size == 0:
        img = np.full((MIN_DISPLAY_DIM_HEATMAP, MIN_DISPLAY_DIM_HEATMAP + COLORBAR_TARGET_WIDTH_PIXELS, 3), (200, 200, 200), dtype=np.uint8)
        return img

    # 1. Scale matrix for display
    display_mat = _get_scaled_matrix_for_display(mat)
    
    # 2. Generate heatmap RGB array
    heatmap_rgb_array, norm_obj, vmin, vmax = _generate_heatmap_rgb_array(
        display_mat, min_val_display, cmap_name, percent
    )

    if not colorbar:
        return heatmap_rgb_array

    if norm_obj is None: # Indicates no valid data in heatmap / dummy image returned
        empty_cbar_space = np.full((heatmap_rgb_array.shape[0], COLORBAR_TARGET_WIDTH_PIXELS, 3), 
                                   (220,220,220), dtype=np.uint8) # Slightly different gray
        return np.hstack((heatmap_rgb_array, empty_cbar_space))


    # 3. Get colorbar ticks and labels
    tick_values, tick_labels = _get_colorbar_ticks_and_labels(vmin, vmax, max_colorbar_ticks, percent)

    # 4. Generate colorbar RGB array
    colorbar_rgb_array = _generate_colorbar_rgb_array(
        norm_obj, cmap_name, tick_values, tick_labels,
        target_height_pixels=heatmap_rgb_array.shape[0],
        font_size_pt=font_size
    )
    
    # 5. Combine heatmap and colorbar
    final_rgb_image = np.hstack((heatmap_rgb_array, colorbar_rgb_array))
    
    return final_rgb_image

def class_distance(classification_weights : torch.Tensor, probability : bool=True):
    classification_weights = classification_weights.cpu().clone().detach()
    classification_weights -= classification_weights.mean(dim=0, keepdim=True)
    classification_weights /= classification_weights.std(dim=0, unbiased=True, keepdim=True)
    class_dmat = torch.cdist(classification_weights, classification_weights)
    if not probability:
        return class_dmat
    class_dmat_cdf = torch.distributions.Chi2(classification_weights.shape[1]).cdf((class_dmat ** 2 / 2))
    if not isinstance(class_dmat_cdf, torch.Tensor):
        raise RuntimeError(f"Unexpected CDF output type {type(class_dmat_cdf)} produced from class distance matrix.")
    return class_dmat_cdf

def plot_model_class_distance(model : nn.Module, **kwargs):
    llw = last_layer_weights(model)
    cdm = class_distance(llw, True).float()
    cdm.fill_diagonal_(torch.nan)
    return plot_heatmap(cdm, **kwargs)