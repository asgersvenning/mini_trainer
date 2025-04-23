from typing import List, Optional

import torch
from torch import nn as nn

from mini_trainer.classifier import get_model


class HierarchicalClassifier(nn.Module):
    def __init__(self, in_features : int, out_features : int, masks : Optional[List[torch.Tensor]]=None, hidden : bool=False):
        super().__init__()
        # Create a BatchNormalization Layer
        self.batch_norm = nn.BatchNorm1d(in_features)

        # Create one hidden layer
        self.hidden = hidden and nn.Linear(in_features, in_features)

        # Create a standard linear layer.
        self.linear = nn.Linear(in_features, out_features, bias=True)

        # Store masks
        self.masks = [m.T.clone(memory_format=torch.contiguous_format) for m in masks] or []
        [m.requires_grad_(False) for m in self.masks]
        
        # Set the bias to -1 and freeze it.
        with torch.no_grad():
            self.linear.bias.fill_(-1)
        self.linear.bias.requires_grad_(False)

    def forward(self, x):
        if self.hidden:
            x = nn.functional.leaky_relu(self.hidden(x), True)
        x = self.batch_norm(x)
        # Compute the normalized log probabilities for the leaf nodes (level 0)
        y0 = nn.functional.log_softmax(self.linear(x), dim = 1)
        ys = [y0]
        # Propagate the probabilities up the hierarchy using the masks
        for mask in self.masks:
            ys.append(nn.functional.log_softmax(torch.logsumexp(ys[-1].unsqueeze(2) + mask, dim = 1), dim=1))
        return ys
    
    @staticmethod
    def load(model_type : str, path : str, masks : Optional[List[torch.Tensor]]=None, device=torch.device("cpu"), dtype=torch.float32):
        # Parse model architecture
        architecture, head_name, _ = get_model(model_type)

        # Read weight file
        weights = torch.load(path, device, weights_only=True)
        if isinstance(weights, dict) and "model" in weights:
            weights = weights["model"]
        num_classes, num_embeddings = weights[f"{head_name}.linear.weight"].shape
        
        # Load weights into model architecture
        setattr(architecture, head_name, HierarchicalClassifier(num_embeddings, num_classes, [m.to(device, dtype) for m in masks]))
        architecture.load_state_dict(weights)
        architecture.to(device, dtype)
        
        return architecture