import torch
from torch import nn
from torch import Tensor


class AdaptiveProbabilisticMatchingLoss(nn.Module):
    """
    Adaptive Probabilistic Matching Loss (APML) for 3D point cloud comparison.
    Computes a differentiable approximation of Earth Mover's Distance using Sinkhorn iterations
    and probabilistic sharpening with sparse matching constraints.
    """

    def __init__(
        self,
        tau: float = 0.01,
        sinkhorn_iters: int = 5,
        eps: float = 1e-5,
        top_k: int = 5,
        sharpness: float = 0.5,
    ):
        super().__init__()
        self.tau = tau
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.top_k = top_k
        self.sharpness = sharpness

    @torch.no_grad()
    def _sinkhorn_iterations(self, sim_matrix: Tensor) -> Tensor:
        """Performs Sinkhorn normalization over a similarity matrix."""
        P = sim_matrix.clone()
        for _ in range(self.sinkhorn_iters):
            P = P / (P.sum(dim=2, keepdim=True) + self.eps)  # Normalize rows
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)  # Normalize columns
        return P

    @torch.no_grad()
    def _sharpen_and_threshold(self, prob_matrix: Tensor) -> Tensor:
        """Applies sharpening and sparsity constraint via top-k filtering."""
        P_sharp = (prob_matrix + self.eps).pow(self.sharpness)

        topk_vals, topk_idx = torch.topk(P_sharp, self.top_k, dim=-1)
        mask = torch.zeros_like(P_sharp).scatter_(-1, topk_idx, 1.0)

        P_filtered = prob_matrix * mask
        P_filtered = P_filtered / (P_filtered.sum(dim=-1, keepdim=True) + self.eps)

        return P_filtered

    def forward(self, pred: Tensor, gt: Tensor) -> Tensor:
        """
        Computes the APML loss between predicted and ground truth point sets.

        Args:
            pred: Tensor of shape [B, N, 3], predicted point cloud
            gt: Tensor of shape [B, M, 3], ground truth point cloud

        Returns:
            Scalar loss value
        """
        dist_matrix = torch.cdist(pred, gt, p=2)  # Shape: [B, N, M]
        sim_matrix = torch.exp(-dist_matrix / self.tau)

        with torch.no_grad():
            P = self._sinkhorn_iterations(sim_matrix)
            P = self._sharpen_and_threshold(P)

        loss = torch.sum(P * dist_matrix, dim=(1, 2)).mean()
        return loss