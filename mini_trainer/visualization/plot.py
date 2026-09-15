import math

import matplotlib as mpl
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.backends import backend_agg
from matplotlib.colors import LogNorm, Normalize
from torch import nn
from torchvision.transforms.functional import resize

from mini_trainer.modeling import class_log_similarity, class_similarity

# --- Constants ---
MIN_DISPLAY_DIM_HEATMAP = 500
MAX_DISPLAY_DIM_HEATMAP = 5000
COLORBAR_RENDER_DPI = 150
COLORBAR_TARGET_WIDTH_PIXELS = 200  # Approximate width for the colorbar image


# --- Helper: Matrix Aggregation ---
def _aggregate_matrix_max(matrix: np.ndarray, block_shape: tuple[int, int], pad_value: float = 0) -> np.ndarray:
    """Aggregates matrix by taking the maximum in blocks.

    Handles non-divisible shapes by padding.
    """
    orig_rows, orig_cols = matrix.shape
    block_rows, block_cols = block_shape

    pad_rows = (block_rows - orig_rows % block_rows) % block_rows
    pad_cols = (block_cols - orig_cols % block_cols) % block_cols

    padded_matrix = matrix
    if pad_rows > 0 or pad_cols > 0:
        padded_matrix = np.pad(matrix, ((0, pad_rows), (0, pad_cols)), mode="constant", constant_values=pad_value)

    new_rows, new_cols = padded_matrix.shape
    target_rows, target_cols = new_rows // block_rows, new_cols // block_cols

    # Efficient reshape and sum for block aggregation
    return padded_matrix.reshape(target_rows, block_rows, target_cols, block_cols).max(axis=3).max(axis=1)


# --- Helper: Matrix Scaling ---
def _get_scaled_matrix_for_display(mat: np.ndarray, pad_value: float = 0) -> np.ndarray:
    """Resizes matrix: downscales then upscales to fit display dimension constraints."""
    processed_mat = mat
    orig_rows, orig_cols = mat.shape

    # 1. Downscale to fit MAX_DISPLAY_DIM_HEATMAP
    block_r = math.ceil(orig_rows / MAX_DISPLAY_DIM_HEATMAP) if orig_rows > MAX_DISPLAY_DIM_HEATMAP else 1
    block_c = math.ceil(orig_cols / MAX_DISPLAY_DIM_HEATMAP) if orig_cols > MAX_DISPLAY_DIM_HEATMAP else 1

    if block_r > 1 or block_c > 1:
        processed_mat = _aggregate_matrix_max(processed_mat, (block_r, block_c), pad_value=pad_value)

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


# --- Helper: Heatmap Array Generation ---
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


def _generate_log_heatmap_rgb_array(display_mat, min_val_display, cmap_name, percent, log_range=None):
    """Map natural-log inputs directly to colours; never exponentiate a matrix."""
    if min_val_display is not None and min_val_display < 0:
        raise ValueError("min_val_display must be nonnegative for log inputs")
    if log_range is not None and (len(log_range) != 2 or not all(math.isfinite(x) for x in log_range) or log_range[0] >= log_range[1]):
        raise ValueError("log_range must contain two finite, increasing natural-log bounds")
    threshold = math.log(min_val_display) if min_val_display is not None and min_val_display > 0 else -np.inf

    def chunks():
        rows = max(1, 262144 // max(1, display_mat.shape[1]))
        for start in range(0, len(display_mat), rows):
            values = np.ma.masked_invalid(display_mat[start : start + rows])
            yield start, np.ma.masked_less_equal(values, threshold)

    minimum, maximum = float("inf"), -float("inf")
    for _, values in chunks():
        if values.count():
            minimum = min(minimum, float(values.min()))
            maximum = max(maximum, float(values.max()))
    rgb = np.zeros((*display_mat.shape, 3), dtype=np.uint8)
    if not np.isfinite(minimum):
        return rgb, None, 0.0, 1.0
    vmin = math.floor(min(-1, minimum / math.log(10))) * math.log(10)
    vmax = 0.0 if percent else maximum
    if vmin >= vmax:
        vmax = vmin + math.log(10)
    if log_range is not None:
        vmin, vmax = log_range
    norm = Normalize(vmin, vmax, clip=log_range is not None)
    cmap = mpl.colormaps[cmap_name].copy()
    cmap.set_bad((0, 0, 0) if cmap_name == "magma" else (1, 1, 1), alpha=1)
    for start, values in chunks():
        rgb[start : start + len(values)] = cmap(norm(values), bytes=True)[:, :, :3]
    return rgb, norm, vmin, vmax


def _log_colorbar_ticks(vmin, vmax, max_ticks, percent):
    """Format log probabilities without underflowing a linear tick value."""
    ticks = np.linspace(vmin, vmax, max(0, max_ticks)).tolist()
    labels = []
    for tick in ticks:
        exponent = tick / math.log(10) + (2 if percent else 0)
        if abs(exponent - round(exponent)) < 1e-6:
            exponent = round(exponent)
        labels.append(f"10^{exponent:g}" + ("%" if percent else ""))
    return ticks, labels


# --- Helper: Colorbar Ticks ---
def _get_colorbar_ticks_and_labels(norm_vmin: float, norm_vmax: float, max_ticks: int, percent: bool) -> tuple[list[float], list[str]]:
    """Generates tick values and labels for the colorbar."""
    if not (norm_vmin > 0 and norm_vmax > 0 and norm_vmin < norm_vmax):
        return [], []

    num_decades = math.log10(norm_vmax / norm_vmin) if norm_vmin > 0 and norm_vmax > 0 else 1
    multipliers = [1, 2, 5] if num_decades < 2 else [1, 1.5, 2, 3, 5, 7]  # Fewer for small ranges

    tick_cands = {norm_vmin, norm_vmax}
    start_exp = math.floor(math.log10(norm_vmin)) if norm_vmin > 0 else 0
    end_exp = math.ceil(math.log10(norm_vmax)) if norm_vmax > 0 else 0

    for exp_val in range(start_exp, end_exp + 1):
        for m in multipliers:
            tick = (10**exp_val) * m
            if norm_vmin <= tick <= norm_vmax:  # Ensure ticks are within actual data range
                tick_cands.add(tick)

    # Filter again to be absolutely sure, then sort
    sorted_ticks = sorted(list(t for t in tick_cands if norm_vmin <= t <= norm_vmax))

    if len(sorted_ticks) > max_ticks:  # Subsample if too many
        indices = np.round(np.linspace(0, len(sorted_ticks) - 1, max_ticks)).astype(int)
        final_ticks = [sorted_ticks[i] for i in sorted(list(set(indices)))]
        # Ensure original vmin and vmax are considered if space allows
        if max_ticks >= 1 and not np.isclose(final_ticks[0], norm_vmin):
            final_ticks.insert(0, norm_vmin)
        if max_ticks >= 2 and not np.isclose(final_ticks[-1], norm_vmax):
            final_ticks.append(norm_vmax)
        final_ticks = sorted(list(set(t for t in final_ticks if norm_vmin <= t <= norm_vmax)))[:max_ticks]
    else:
        final_ticks = sorted_ticks

    # Ensure at least two ticks (min/max) if possible, if list became empty by max_ticks=0 or 1
    if not final_ticks and len(sorted_ticks) >= 1:
        final_ticks = [sorted_ticks[0]]
        if len(sorted_ticks) > 1:
            final_ticks.append(sorted_ticks[-1])
        final_ticks = sorted(list(set(final_ticks)))

    labels = []
    for v_tick in final_ticks:
        val_fmt = v_tick * 100 if percent else v_tick
        lab_str = ""
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


# --- Helper: Colorbar Array Generation ---
def _generate_colorbar_rgb_array(
    norm_obj: mpl.colors.Normalize,
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


# --- Main Plotting Function ---
def plot_heatmap(
    mat: np.ndarray | torch.Tensor,
    cmap_name: str = "magma",
    font_size: int = 10,  # For colorbar labels
    max_colorbar_ticks: int = 8,
    percent: bool = True,
    min_val_display: float | None = None,
    colorbar: bool = True,
    *,
    log_input: bool = False,
    log_range: tuple[float, float] | None = None,
):
    """Plots a high-resolution confusion matrix using NumPy and Matplotlib.

    Returns a combined RGB NumPy array (heatmap + colorbar).
    With log_input=True, inputs are natural logs and stay logarithmic through
    aggregation, normalization and colourbar formatting. min_val_display remains
    a linear threshold. -inf represents zero and is masked; padding uses -inf.
    log_range optionally clips colours to natural-log bounds, not input values.
    """
    if isinstance(mat, torch.Tensor):
        mat = mat.cpu().detach().float().numpy()

    if mat.size == 0:
        img = np.full((MIN_DISPLAY_DIM_HEATMAP, MIN_DISPLAY_DIM_HEATMAP + COLORBAR_TARGET_WIDTH_PIXELS, 3), (200, 200, 200), dtype=np.uint8)
        return img

    # 1. Scale matrix for display
    display_mat = _get_scaled_matrix_for_display(mat, pad_value=-np.inf if log_input else 0)

    # 2. Generate heatmap RGB array
    if log_range is not None and not log_input:
        raise ValueError("log_range requires log_input=True")
    if log_input:
        heatmap_rgb_array, norm_obj, vmin, vmax = _generate_log_heatmap_rgb_array(
            display_mat, min_val_display, cmap_name, percent, log_range
        )
    else:
        heatmap_rgb_array, norm_obj, vmin, vmax = _generate_heatmap_rgb_array(display_mat, min_val_display, cmap_name, percent)

    if not colorbar:
        return heatmap_rgb_array

    if norm_obj is None:  # Indicates no valid data in heatmap / dummy image returned
        empty_cbar_space = np.full(
            (heatmap_rgb_array.shape[0], COLORBAR_TARGET_WIDTH_PIXELS, 3), (220, 220, 220), dtype=np.uint8
        )  # Slightly different gray
        return np.hstack((heatmap_rgb_array, empty_cbar_space))

    # 3. Get colorbar ticks and labels
    tick_generator = _log_colorbar_ticks if log_input else _get_colorbar_ticks_and_labels
    tick_values, tick_labels = tick_generator(vmin, vmax, max_colorbar_ticks, percent)
    if log_range is not None and tick_labels:
        tick_labels[0] = "≤" + tick_labels[0]
        if len(tick_labels) > 1:
            tick_labels[-1] = "≥" + tick_labels[-1]

    # 4. Generate colorbar RGB array
    colorbar_rgb_array = _generate_colorbar_rgb_array(
        norm_obj, cmap_name, tick_values, tick_labels, target_height_pixels=heatmap_rgb_array.shape[0], font_size_pt=font_size
    )

    # 5. Combine heatmap and colorbar
    final_rgb_image = np.hstack((heatmap_rgb_array, colorbar_rgb_array))

    return final_rgb_image


def plot_class_distance_matrix(model: nn.Module, *, log_domain: bool = False, **kwargs):
    """Plot the pairwise class distance matrix.

    CDF of pairwise inner products between rows
    in the last-layer weight matrix, assuming that these
    have unit norm.
    log_domain=True computes the same complementary probability directly in
    float32 log space, avoiding subtraction from a rounded CDF. The default
    preserves the legacy rendering for comparisons.
    """
    if log_domain:
        return [plot_heatmap(log_tail.cpu(), log_input=True, **kwargs) for log_tail in class_log_similarity(model, complement=True)]
    return [plot_heatmap(1 - csi.cpu(), **kwargs) for csi in class_similarity(model, cdf=True)]
