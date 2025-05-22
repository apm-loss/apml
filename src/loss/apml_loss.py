import torch
from torch import nn, Tensor


class AdaptiveProbabilisticMatchingLoss(nn.Module):
    """
    Adaptive Probabilistic Matching Loss (APML) for 3D point cloud comparison.
    Approximates optimal transport using a temperature-scaled softmin that ensures
    a minimum assignment probability, followed by Sinkhorn normalization.
    """

    def __init__(self, min_softmax_value: float = 0.8, eps: float = 1e-8):
        super().__init__()
        self.min_softmax_value = min_softmax_value
        self.eps = eps

    @torch.no_grad()
    def _softmin_with_min_probability(self, cost: Tensor, dim: int = 2, margin: float = 1e-8) -> Tensor:
        """
        Compute a temperature-scaled softmin across the specified dimension,
        with temperature analytically derived to guarantee a minimum probability.
        """
        if dim != 2:
            cost = cost.transpose(dim, 2)

        B, N, M = cost.shape
        flat = cost.reshape(-1, M)

        min_vals = flat.min(dim=1, keepdim=True).values
        flat = flat - min_vals

        top2_vals, _ = flat.topk(2, dim=1, largest=False)
        gaps = top2_vals[:, 1] + margin
        T = -torch.log(torch.tensor((1 - self.min_softmax_value) / (M - 1), device=cost.device, dtype=cost.dtype)) / gaps

        scaled = -flat * T[:, None]
        softmaxed = torch.softmax(scaled, dim=1)

        all_equal = (top2_vals[:, 1] == 0)
        if all_equal.any():
            uniform = torch.full((M,), 1.0 / M, device=cost.device, dtype=cost.dtype)
            softmaxed[all_equal] = uniform

        out = softmaxed.view(B, N, M)
        if dim != 2:
            out = out.transpose(2, dim)

        return out

    @torch.no_grad()
    def _sinkhorn(self, P: Tensor, iterations: int = 20) -> Tensor:
        """Performs Sinkhorn normalization on a soft assignment matrix."""
        for _ in range(iterations):
            P = P / (P.sum(dim=2, keepdim=True) + self.eps)
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)
        return P

    def forward(self, predicted: Tensor, ground_truth: Tensor) -> Tensor:
        """
        Compute the Adaptive Probabilistic Matching Loss between predicted and ground truth point clouds.

        Args:
            predicted: Tensor of shape [B, N, 3]
            ground_truth: Tensor of shape [B, M, 3]

        Returns:
            Scalar loss value
        """
        cost = torch.cdist(predicted.float(), ground_truth.float(), p=2)  # [B, N, M]

        with torch.no_grad():
            P1 = self._softmin_with_min_probability(cost, dim=2)
            P2 = self._softmin_with_min_probability(cost, dim=1)
            P = 0.5 * (P1 + P2)
            P = self._sinkhorn(P)

        loss = torch.sum(P * cost, dim=(1, 2)).mean()
        return loss
