from typing import List, Optional

import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.cluster.hierarchy import dendrogram, linkage

from radialtree import radialTreee
from radialtree_helpers import multidimensional_distance_probability
from scipy.special import betainc


def hierarchical_cluster_llw(last_layer_weights : torch.Tensor, labels : Optional[List[str]]):
    last_layer_weights = last_layer_weights.clone(memory_format=torch.contiguous_format)

    # last_layer_weights -= last_layer_weights.mean(dim=0, keepdim=True)
    # last_layer_weights /= last_layer_weights.std(dim=0, keepdim=True)
    
    E = last_layer_weights.shape[-1]

    cos_mat = (1 - torch.corrcoef(last_layer_weights))/2
    eucl_mat = torch.cdist(last_layer_weights, last_layer_weights)
    eucl_mat = multidimensional_distance_probability(eucl_mat, E)
    cor_mat = betainc((E - 1)/2, (E - 1)/2, cos_mat)

    fig, axs = plt.subplots(1, 3, figsize = (13, 5))

    maxs0 = axs[0].matshow(cos_mat.cpu().fill_diagonal_(torch.nan))
    divider0 = make_axes_locatable(axs[0])
    cax0 = divider0.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(maxs0, cax0, axs[0])
    axs[0].set_title("Cosine similarity")

    maxs1 = axs[1].matshow(eucl_mat.cpu().fill_diagonal_(torch.nan))
    divider1 = make_axes_locatable(axs[1])
    cax1 = divider1.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(maxs1, cax1, axs[1])
    axs[1].set_title("Euclidean distance")

    maxs2 = axs[2].matshow(cor_mat.cpu().fill_diagonal_(torch.nan))
    divider2 = make_axes_locatable(axs[2])
    cax2 = divider2.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(maxs2, cax2, axs[2])
    axs[2].set_title("Correlation")

    plt.show()

    n_display_tree = len(cos_mat) // 1

    mat_short = cor_mat[:n_display_tree, :n_display_tree].cpu()
    fdmat = mat_short[*torch.triu_indices(*mat_short.shape, offset=1)]
    # Remove "head" i.e. minimum distance should be close to 0
    # mi, ma = fdmat.min().item(), fdmat.max().item()
    # fdmat = fdmat - mi + (ma - mi) * 0.01
    clust = linkage(fdmat, "ward")

    dendr = dendrogram(clust, labels=labels, orientation="left", color_threshold=0.05, no_plot=True)

    fig = plt.figure(figsize=(20, 20), dpi=200)
    fig.gca().set_aspect(1)
    radialTreee(dendr, ax=fig.gca(), pallete="nipy_spectral", fontsize=4, sample_classes={"cluster" : [c for _, c in sorted(zip(dendr["leaves"], dendr["leaves_color_list"]))]})

    plt.show()

    return dendr["leaves"]