import math

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends import backend_agg
from matplotlib.colors import LogNorm
from torch import nn
from torchvision.transforms.functional import resize

from mini_trainer.modeling import class_similarity

MIN_DISPLAY_DIM_HEATMAP = 500
MAX_DISPLAY_DIM_HEATMAP = 5000
COLORBAR_RENDER_DPI = 150
COLORBAR_TARGET_WIDTH_PIXELS = 200  # Approximate width for the colorbar image


def _aggregate_matrix_max(matrix: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    """Aggregates matrix by taking the maximum in blocks.

    Handles non-divisible shapes by padding.
    """
    orig_rows, orig_cols = matrix.shape
    block_rows, block_cols = block_shape

    pad_rows = (block_rows - orig_rows % block_rows) % block_rows
    pad_cols = (block_cols - orig_cols % block_cols) % block_cols

    padded_matrix = matrix
    if pad_rows > 0 or pad_cols > 0:
        padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=0)

    new_rows, new_cols = padded_matrix.shape
    target_rows, target_cols = new_rows // block_rows, new_cols // block_cols

    return padded_matrix.reshape(target_rows, block_rows, target_cols, block_cols).max(axis=3).max(axis=1)


def _get_scaled_matrix_for_display(mat: np.ndarray) -> np.ndarray:
    """Resizes matrix: downscales then upscales to fit display dimension constraints."""
    processed_mat = mat
    orig_rows, orig_cols = mat.shape

    # 1. Downscale to fit MAX_DISPLAY_DIM_HEATMAP
    block_r = math.ceil(orig_rows / MAX_DISPLAY_DIM_HEATMAP) if orig_rows > MAX_DISPLAY_DIM_HEATMAP else 1
    block_c = math.ceil(orig_cols / MAX_DISPLAY_DIM_HEATMAP) if orig_cols > MAX_DISPLAY_DIM_HEATMAP else 1

    if block_r > 1 or block_c > 1:
        processed_mat = _aggregate_matrix_max(processed_mat, (block_r, block_c))

    curr_rows, curr_cols = processed_mat.shape

    # 2. Upscale to meet MIN_DISPLAY_DIM_HEATMAP, constrained by MAX_DISPLAY_DIM_HEATMAP
    k_ideal = max(
        math.ceil(MIN_DISPLAY_DIM_HEATMAP / curr_rows) if curr_rows > 0 and curr_rows < MIN_DISPLAY_DIM_HEATMAP else 1,
        math.ceil(MIN_DISPLAY_DIM_HEATMAP / curr_cols) if curr_cols > 0 and curr_cols < MIN_DISPLAY_DIM_HEATMAP else 1,
    )
    k_cap = min(
        math.floor(MAX_DISPLAY_DIM_HEATMAP / curr_rows) if curr_rows > 0 else float("inf"),
        math.floor(MAX_DISPLAY_DIM_HEATMAP / curr_cols) if curr_cols > 0 else float("inf"),
    )
    final_k = int(max(1, min(k_ideal, k_cap)))

    if final_k > 1:
        return np.kron(processed_mat, np.ones((final_k, final_k), dtype=processed_mat.dtype))

    return processed_mat.copy() if processed_mat is mat else processed_mat


def _generate_heatmap_rgb_array(display_mat: np.ndarray, min_val_display: float | None, cmap_name: str, percent: bool):
    """Generates the RGB heatmap image array using Matplotlib colormaps, and returns norm info."""

    # Scan and map bounded row chunks: avoid full float64 RGBA/masked copies.
    def chunks():
        rows = max(1, 262144 // max(1, display_mat.shape[1]))
        for start in range(0, display_mat.shape[0], rows):
            values = np.ma.masked_invalid(np.asarray(display_mat[start : start + rows], dtype=float))
            if min_val_display is not None:
                values = np.ma.masked_less_equal(values, min_val_display)
            yield start, values

    minimum, maximum = float("inf"), -float("inf")
    for _, values in chunks():
        positive = values.compressed()
        positive = positive[positive > 0]
        if positive.size:
            minimum = min(minimum, float(positive.min()))
            maximum = max(maximum, float(positive.max()))
    rgb = np.zeros((*display_mat.shape, 3), dtype=np.uint8)
    if not np.isfinite(minimum):
        return rgb, None, 0.0, 1.0
    norm_vmin = 10 ** math.floor(math.log10(min(0.1, minimum)))
    norm_vmax = 1.0 if percent else maximum
    if norm_vmin >= norm_vmax:
        norm_vmax = norm_vmin * (1.1 if norm_vmin > 1 else 2.0)
    norm = LogNorm(vmin=norm_vmin, vmax=norm_vmax)
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad(color=(0, 0, 0) if cmap_name == "magma" else (1, 1, 1), alpha=1.0)
    for start, values in chunks():
        rgb[start : start + len(values)] = cmap(norm(values), bytes=True)[:, :, :3]
    return rgb, norm, norm_vmin, norm_vmax


def _get_colorbar_ticks_and_labels(norm_vmin: float, norm_vmax: float, max_ticks: int, percent: bool) -> tuple[list[float], list[str]]:
    """Generates tick values and labels for the colorbar."""
    if not (norm_vmin > 0 and norm_vmax > 0 and norm_vmin < norm_vmax):
        return [], []

    num_decades = math.log10(norm_vmax / norm_vmin)
    multipliers = [1, 2, 5] if num_decades < 2 else [1, 1.5, 2, 3, 5, 7]  # Fewer for small ranges

    tick_cands = {norm_vmin, norm_vmax}
    start_exp = math.floor(math.log10(norm_vmin))
    end_exp = math.ceil(math.log10(norm_vmax))

    for exp_val in range(start_exp, end_exp + 1):
        for m in multipliers:
            tick = (10**exp_val) * m
            if norm_vmin <= tick <= norm_vmax:  # Ensure ticks are within actual data range
                tick_cands.add(tick)

    final_ticks = sorted(tick_cands)
    if max_ticks == 0:
        # Preserve the legacy zero-limit fallback to both bounds.
        final_ticks = [norm_vmin, norm_vmax]
    elif len(final_ticks) > max_ticks:
        indices = np.round(np.linspace(0, len(final_ticks) - 1, max_ticks)).astype(int)
        final_ticks = [final_ticks[i] for i in indices]

    labels = []
    for v_tick in final_ticks:
        val_fmt = v_tick * 100 if percent else v_tick
        if percent:
            if abs(val_fmt) < 0.01 and val_fmt != 0:
                lab_str = f"{val_fmt:.1e}%"
            elif np.isclose(val_fmt, round(val_fmt)):
                lab_str = f"{int(round(val_fmt))}%"
            else:
                lab_str = f"{val_fmt:.2g}%"
        else:  # Non-percent
            if abs(v_tick) >= 1000 or (abs(v_tick) < 0.001 and v_tick != 0):
                lab_str = f"{v_tick:.2g}"
            else:
                lab_str = f"{v_tick:.3g}".rstrip("0").rstrip(".")  # General, remove trailing .0
        labels.append(lab_str)

    return final_ticks, labels


def _generate_colorbar_rgb_array(
    norm_obj: mpl.colors.LogNorm,
    cmap_name_str: str,
    tick_list: list[float],
    tick_label_list: list[str],
    target_height_pixels: int,
    font_size_pt: int,
) -> np.ndarray:
    """Renders a colorbar using Matplotlib to an RGB NumPy array of target_height_pixels."""
    fig_width_inches = COLORBAR_TARGET_WIDTH_PIXELS / COLORBAR_RENDER_DPI
    fig_height_inches = target_height_pixels / COLORBAR_RENDER_DPI

    fig_cbar = mpl.figure.Figure(figsize=(fig_width_inches, fig_height_inches), dpi=COLORBAR_RENDER_DPI)
    canvas_cbar = backend_agg.FigureCanvasAgg(fig_cbar)

    ax_cbar_rect = [0.15, 0.05, 0.3, 0.9]  # [left, bottom, width_of_strip, height_of_strip]
    ax_cbar = fig_cbar.add_axes(ax_cbar_rect)
    cmap_obj_for_cbar = mpl.colormaps[cmap_name_str] if isinstance(cmap_name_str, str) else cmap_name_str

    cb = mpl.colorbar.ColorbarBase(ax_cbar, cmap=cmap_obj_for_cbar, norm=norm_obj, orientation="vertical", ticks=tick_list)

    if tick_list:  # Only set labels if there are ticks
        cb.set_ticklabels(tick_label_list)
        cb.ax.tick_params(labelsize=font_size_pt, length=0)  # No tick lines
        cb.outline.set_visible(False)
    else:  # No ticks, make it a plain strip
        ax_cbar.set_axis_off()

    canvas_cbar.draw()  # Render the figure
    img_rgb = np.asarray(canvas_cbar.buffer_rgba())[:, :, :3]  # Get buffer and convert to ndarray

    if img_rgb.shape[0] != target_height_pixels:
        img_rgb = (
            resize(
                torch.tensor(img_rgb).permute(2, 0, 1),
                [target_height_pixels, int(round(target_height_pixels / img_rgb.shape[0] * img_rgb.shape[1]))],
            )
            .permute(1, 2, 0)
            .numpy()
        )

    plt.close(fig_cbar)
    return img_rgb


def plot_heatmap(
    mat: np.ndarray | torch.Tensor,
    cmap_name: str = "magma",
    font_size: int = 10,  # For colorbar labels
    max_colorbar_ticks: int = 8,
    percent: bool = True,
    min_val_display: float | None = None,
    colorbar: bool = True,
):
    """Plots a high-resolution confusion matrix using NumPy and Matplotlib.

    Return an RGB uint8 array with an optional colorbar; empty input gives a gray image.
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().detach().float().numpy()

    if mat.size == 0:
        img = np.full((MIN_DISPLAY_DIM_HEATMAP, MIN_DISPLAY_DIM_HEATMAP + COLORBAR_TARGET_WIDTH_PIXELS, 3), (200, 200, 200), dtype=np.uint8)
        return img

    display_mat = _get_scaled_matrix_for_display(mat)

    heatmap_rgb_array, norm_obj, vmin, vmax = _generate_heatmap_rgb_array(display_mat, min_val_display, cmap_name, percent)

    if not colorbar:
        return heatmap_rgb_array

    if norm_obj is None:  # Indicates no valid data in heatmap / dummy image returned
        empty_cbar_space = np.full(
            (heatmap_rgb_array.shape[0], COLORBAR_TARGET_WIDTH_PIXELS, 3), (220, 220, 220), dtype=np.uint8
        )  # Slightly different gray
        return np.hstack((heatmap_rgb_array, empty_cbar_space))

    tick_values, tick_labels = _get_colorbar_ticks_and_labels(vmin, vmax, max_colorbar_ticks, percent)

    colorbar_rgb_array = _generate_colorbar_rgb_array(
        norm_obj, cmap_name, tick_values, tick_labels, target_height_pixels=heatmap_rgb_array.shape[0], font_size_pt=font_size
    )

    return np.hstack((heatmap_rgb_array, colorbar_rgb_array))


def plot_class_distance_matrix(model: nn.Module, **kwargs):
    """Plot the pairwise class distance matrix.

    CDF of pairwise inner products between rows
    in the last-layer weight matrix, assuming that these
    have unit norm.
    """
    return [plot_heatmap(1 - csi.cpu(), **kwargs) for csi in class_similarity(model, cdf=True)]
