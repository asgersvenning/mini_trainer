import colorsys
import csv
import os
from collections import Counter, defaultdict
from typing import Callable, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

name_fmt = "JPGImages{ImageLabel_noext}.{IndividualNumber}.jpg"
def image_basename(idx : int, data : Dict):
    ImageLabel = str(data["ImageLabel"][idx])
    IndividualNumber = str(data["IndividualNumber"][idx])

    ImageLabel_noext = ImageLabel.removesuffix(".tif")

    return name_fmt.format(ImageLabel_noext = ImageLabel_noext, IndividualNumber = IndividualNumber)

def check_split(species : str, basename : str, root : str):
    train_path = os.path.join(root, "Images", "training", species, basename)
    testing_path = os.path.join(root, "Images", "testing", species, basename)
    zero_path = os.path.join(root, "Images", "zero", species, basename)
    if os.path.exists(train_path):
        return "train"
    if os.path.exists(testing_path):
        return "test"
    if os.path.exists(zero_path):
        return "zero"
    return "unknown"

def get_path(idx : int, data : Dict, root : str):
    species = str(data["SpeciesName"][idx])
    basename = image_basename(idx, data)
    split = check_split(species, basename, root)
    if split == "unknown":
        raise FileNotFoundError(f"Unable to construct path for image ({idx}) in unknown split")
    if split in ["test", "train"]:
        split += "ing"
    return os.path.abspath(os.path.join(root, "Images", split, species, basename))

def attempt_convert(l : List, new : List[Callable] = [int, float], exceptions : List[str]=["NULL", "NA", "NaN", "NA/NA", ""]) -> List:
    for typ in new:
        def f(x):
            x = x.strip()
            if x in exceptions:
                return np.nan
            else:
                return typ(x)
        try:
            return list(map(f, l))
        except:
            continue
    return l

def read_data(dir : str, file : str="Carabid_All.csv", verbose : bool=True):
    with open(os.path.join(dir, file)) as f:
        reader = csv.reader(f)

        fields = next(reader)

        image_data = defaultdict(list)

        for row in reader:
            for c, e in zip(fields, row):
                image_data[c].append(e)

    image_data = {n : np.array(attempt_convert(d)) for n, d in image_data.items()}

    image_data["split"] = [np.array(check_split(sp, image_basename(i, image_data), dir)) for i, sp in enumerate(image_data["SpeciesName"])]
    if verbose:
        print("Data split:", Counter(map(str, image_data["split"])))

    image_data = {k : np.compress(image_data["split"] != np.array("unknown"), v) for k, v in image_data.items()}
    image_data["path"] = np.array([get_path(i, image_data, dir) for i in range(len(image_data["split"]))])

    if verbose:
        print("All paths exist:", all(map(os.path.exists, image_data["path"])))

    return image_data

def skew(arr):
    """
    Calculate the skewness of a numeric NumPy array using only numpy.

    Args:
        arr (numpy.ndarray): Input array.

    Returns:
        float: Skewness of the array.
    """
    # Remove NaN values
    arr = arr[~np.isnan(arr)]

    # Calculate the mean and standard deviation
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)  # ddof=0 for population standard deviation

    # Calculate the third moment (numerator of skewness)
    third_moment = np.mean((arr - mean) ** 3)

    # Calculate skewness
    skewness = third_moment / (std ** 3)

    return skewness

def calculate_skewness(arr):
    """
    Calculate the skewness of a numeric NumPy array using only numpy.
    """
    arr = arr[~np.isnan(arr)]
    mean = np.mean(arr)
    std = np.std(arr, ddof=0)
    third_moment = np.mean((arr - mean) ** 3)
    skewness = third_moment / (std ** 3)
    return skewness

def apply_transformations(arr, depth=2, current_depth=0, desc=None):
    """
    Recursively apply transformations to the array and return the transformed array
    with the lowest absolute skewness, along with the transformation names.

    Args:
        arr (numpy.ndarray): Input array.
        depth (int): Maximum depth of nested transformations.
        current_depth (int): Current depth of recursion.

    Returns:
        numpy.ndarray: Transformed array.
        str: Description of transformations applied.
        float: Skewness of the transformed array.
    """
    transformations = {
        "log": np.log(arr) if np.all(arr > 0) else None,
        "shift-log": np.log(arr - np.min(arr) + 1),
        "log1p": np.log1p(arr) if np.all(arr >= 0) else None,
        "sqrt": np.sqrt(arr) if np.all(arr >= 0) else None,
        "inverse": 1 / arr if np.all(arr != 0) else None
    }

    # Initialize best transformation
    best_skewness = calculate_skewness(arr)
    best_transformed_arr = arr
    desc = desc or "original"
    best_transformation_desc = desc

    # Try each transformation
    for name, transformed_arr in transformations.items():
        if name == "inverse" and "inverse" in best_transformation_desc:
            continue
        if transformed_arr is not None:
            skewness = calculate_skewness(transformed_arr)

            # If current transformation improves skewness, update best result
            if abs(skewness) < abs(best_skewness):
                best_skewness = skewness
                best_transformed_arr = transformed_arr
                best_transformation_desc = name

            # If depth allows, recursively apply transformations
            if current_depth < depth - 1:
                nested_arr, nested_desc, nested_skewness = apply_transformations(
                    transformed_arr, depth, current_depth + 1, f"{desc} -> {name}"
                )
                if abs(nested_skewness) < abs(best_skewness):
                    best_skewness = nested_skewness
                    best_transformed_arr = nested_arr
                    best_transformation_desc = nested_desc

    return best_transformed_arr, best_transformation_desc, best_skewness

def normalize_array(arr, transform_threshold=1.0, check_transform=True, depth=1, verbose=False):
    """
    Normalize a numeric NumPy array by subtracting the mean and dividing by the standard deviation.
    Handles np.nan values by ignoring them during computation. Additionally, checks if the array
    is better described by a log-, sqrt-, or inverse-normal distribution and applies the transformation
    that reduces skewness the most, if it improves the distribution.

    Args:
        arr (numpy.ndarray): Input array.
        transform_threshold (float): Threshold for skewness to decide if a transformation is needed.
        check_transform (bool): Whether to check and apply transformations.

    Returns:
        numpy.ndarray: Normalized array, or the original array if it is not numeric.
    """
    # Check if the array is numeric (int or float)
    if not np.issubdtype(arr.dtype, np.number):
        if verbose:
            print("Input array is not numeric. Returning the original array.")
        return arr

    if check_transform:
        # Compute skewness of the original array
        original_skewness = calculate_skewness(arr)

        # Apply transformations and select the best one
        if abs(original_skewness) > transform_threshold:
            transformed_arr, best_transformation, best_skewness = apply_transformations(arr, depth=depth)
            if best_transformation != "original" and verbose:
                print(f"Applied {best_transformation} transformation. Skewness reduced from {original_skewness:.2f} to {best_skewness:.2f}.")
        else:
            transformed_arr = arr
            if verbose:
                print("No transformation applied. Skewness is within the threshold.")
    else:
        transformed_arr = arr

    # Compute mean and standard deviation, ignoring np.nan values
    mean = np.nanmean(transformed_arr)
    std = np.nanstd(transformed_arr)

    # Avoid division by zero in case of zero standard deviation
    if std == 0:
        if verbose:
            print("Standard deviation is zero. Returning the original array.")
        return arr

    # Normalize the array
    normalized_arr = (transformed_arr - mean) / std

    return normalized_arr

def multidimensional_distance_probability(x, d):
    return torch.distributions.Chi2(df=d).cdf(x**2 / 2)

def normalize_multidimensional_distance(x, d):
    return torch.distributions.Normal(0, 1).icdf(torch.distributions.Chi2(df=d).cdf(x**2 / 2))

def high_contrast(rgb):
    # Convert RGB to HLS (Hue, Lightness, Saturation)
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # Compute the complementary hue (shift by 180°)
    comp_h = (h + 0.5) % 1.0
    # Invert lightness for contrast: if the original is dark, use light; if light, use dark.
    comp_l = 1 - l
    # Boost saturation for a more vivid contrasting color.
    comp_s = max(s, 0.5)
    # Convert the complementary HLS back to RGB
    return colorsys.hls_to_rgb(comp_h, comp_l, comp_s)

def connect_points(points1, points2, ax=None, autoscale=True, **kwargs):
    """
    Connect corresponding points from two sets with line segments on a Matplotlib Axes.

    Parameters:
        points1 : array-like, shape (n, 2)
            First set of (x, y) points.
        points2 : array-like, shape (n, 2)
            Second set of (x, y) points.
        ax : matplotlib.axes.Axes, optional
            The axes to which the segments will be added. Uses current axes (plt.gca()) if None.
        autoscale : bool, default True
            Whether to adjust the view limits to include the segments.
        **kwargs:
            Additional keyword arguments passed to the LineCollection (e.g., colors, linewidths).

    Returns:
        lc : matplotlib.collections.LineCollection
            The created LineCollection object.
    """
    if ax is None:
        ax = plt.gca()
    
    # Ensure inputs are NumPy arrays and have the proper shape.
    points1 = np.asarray(points1)
    points2 = np.asarray(points2)
    
    if points1.shape != points2.shape:
        raise ValueError("points1 and points2 must have the same shape.")
    
    # Each segment is a pair of points, so stack along a new axis.
    segments = np.stack((points1, points2), axis=1)
    
    # Create the LineCollection.
    lc = LineCollection(segments, **kwargs)
    
    # Add the LineCollection to the provided axes.
    ax.add_collection(lc)
    
    # Optionally autoscale the view to include the new segments.
    if autoscale:
        ax.autoscale_view()
    
    return lc

def connect_many_to_one(points, target, ax=None, autoscale=True, **kwargs):
    """
    Connect many points to a single target point with line segments on a Matplotlib Axes.

    Parameters:
        points : array-like, shape (n, 2)
            An array of (x, y) points.
        target : array-like, shape (2,)
            The target (x, y) point to which all points will be connected.
        ax : matplotlib.axes.Axes, optional
            The axes to which the segments will be added. Defaults to plt.gca() if None.
        autoscale : bool, default True
            Whether to adjust the view limits to include all the segments.
        **kwargs:
            Additional keyword arguments passed to the LineCollection (e.g., colors, linewidths).

    Returns:
        lc : matplotlib.collections.LineCollection
            The created LineCollection object.
    """
    if ax is None:
        ax = plt.gca()
    
    # Convert inputs to numpy arrays.
    points = np.asarray(points)
    target = np.asarray(target)
    
    # Ensure target is a 2D point.
    if target.shape != (2,):
        raise ValueError("The target must be a 2-element array-like representing (x, y).")
    
    # Repeat the target point to match the number of points.
    targets = np.tile(target, (points.shape[0], 1))
    
    # Each segment is formed by pairing a point with the target.
    segments = np.stack((points, targets), axis=1)
    
    # Create and add the LineCollection.
    lc = LineCollection(segments, **kwargs)
    ax.add_collection(lc)
    
    # Optionally autoscale the axes view.
    if autoscale:
        ax.autoscale_view()
    
    return lc

def first_non_nan(x, dim):
    """
    Returns the first non-NaN element along the given dimension of tensor x.
    If all elements along that slice are NaN, returns NaN for that slice.

    Parameters:
        x (torch.Tensor): Input tensor that may contain NaNs.
        dim (int): Dimension along which to search for the first non-NaN element.

    Returns:
        torch.Tensor: Tensor with the dimension `dim` reduced, containing the first non-NaN
                      element from each slice along that dimension.
    """
    # Create a boolean mask where True indicates non-NaN elements.
    mask = ~torch.isnan(x)
    
    # Determine for each slice whether any non-NaN value exists.
    valid = mask.any(dim=dim)
    
    # For each slice, find the index of the first True value.
    # Note: if a slice contains only False, argmax returns 0.
    first_idx = mask.float().argmax(dim=dim)
    
    # Prepare indices for torch.gather: first_idx needs to have the same number of dims as x.
    gather_idx = first_idx.unsqueeze(dim)
    
    # Gather the first candidate along the specified dimension.
    result = torch.gather(x, dim, gather_idx).squeeze(dim)
    
    # For slices with no non-NaN element, overwrite the result with NaN.
    result = torch.where(valid, result, torch.full_like(result, float('nan')))
    
    return result