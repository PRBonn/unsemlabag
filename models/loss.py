import torch
import torch.nn as nn


def get_kl_divergence_prior_loss(alpha: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

    alpha_kl = targets + (1 - targets) * alpha
    alpha_kl_sum = torch.sum(alpha_kl, dim=1, keepdim=True)
    ones = torch.ones_like(alpha)
    kl_log_term = (
        torch.lgamma(alpha_kl_sum)
        - torch.lgamma(torch.sum(ones, dim=1, keepdim=True))
        - torch.sum(torch.lgamma(alpha_kl), dim=1, keepdim=True)
    )
    kl_digamma_term = torch.sum(
        (alpha_kl - 1) * (torch.digamma(alpha_kl) - torch.digamma(alpha_kl_sum)), dim=1, keepdim=True
    )
    return (kl_log_term + kl_digamma_term).squeeze(dim=1)


class CrossEntropyBayesRiskLoss(nn.Module):
    def __init__(self, reduction: str = "mean", ignore_index: int = 3, weights=[1.0, 1.0, 1.0]):
        super(CrossEntropyBayesRiskLoss, self).__init__()

        self.reduction = reduction
        self.ignore_index = ignore_index
        self.weights = weights

    def forward(self, evidence: torch.Tensor, targets: torch.Tensor, kl_div_coeff: float) -> torch.Tensor:
        targets_idx = torch.argmax(targets, dim=1)
        msk = (targets_idx != self.ignore_index).float()

        weights_mask = torch.zeros_like(msk)
        weights_mask[targets_idx == 0] = self.weights[0]
        weights_mask[targets_idx == 1] = self.weights[1]
        weights_mask[targets_idx == 2] = self.weights[2]

        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)
        xentropy_bayes_risk_loss = torch.sum(targets * (torch.digamma(S) - torch.digamma(alpha)), dim=1)

        kl_div_prior_loss = get_kl_divergence_prior_loss(alpha, targets)
        loss = (xentropy_bayes_risk_loss + kl_div_coeff * kl_div_prior_loss) * msk

        loss = (loss * weights_mask).sum() / msk.sum()
        return loss
