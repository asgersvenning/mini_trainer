"""Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
From: https://github.com/HobbitLong/SupContrast/blob/master/losses.py
"""

import torch
import torch.nn as nn

from mini_trainer.classifier import EmbeddingContext, SupervisionContext


class SupConLoss(nn.modules.loss._Loss):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR.
    """

    def __init__(  # noqa: D107
        self, temperature: float = 0.07, base_temperature: float = 0.07
    ):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self):
        """Compute contrastive loss for model.
        If both `labels` and `mask` are None, it degenerates to SimCLR unsupervised loss:
            https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """
        if not EmbeddingContext.active():
            raise RuntimeError("Contrastive loss cannot be computed when `EmbeddingContext` is not active.")
        features = EmbeddingContext.get()
        if features.ndim == 2:
            features = features.unsqueeze(1)
        assert features is not None
        labels = SupervisionContext.get()
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError("`features` needs to be [bsz, n_views, ...],at least 3 dimensions are required")
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(contrast_feature @ contrast_feature.T, self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(contrast_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0)
        mask = mask * logits_mask

        # # compute log_prob
        # exp_logits = torch.exp(logits) * logits_mask
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        log_prob = logits - torch.logsumexp(logits + 1e-9 * (1 - mask), dim=1, keepdim=True)

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss
