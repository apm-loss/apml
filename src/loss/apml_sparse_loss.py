import torch
from torch.autograd import Function
import apml_sparse


class APMLSparseFunction(Function):
    @staticmethod
    def forward(ctx, x, y, p_min=0.8, threshold=1e-10):
        COO_i, COO_j, COO_values, loss = apml_sparse.forward(
            x.contiguous(), y.contiguous(),
            float(p_min), float(threshold)
        )
        ctx.save_for_backward(x, y, COO_i, COO_j, COO_values)
        return loss[0]

    @staticmethod
    def backward(ctx, grad_output):
        x, y, COO_i, COO_j, COO_values = ctx.saved_tensors
        B, N, D = x.shape
        M = y.shape[1]

        grad_x = torch.zeros_like(x)
        grad_y = torch.zeros_like(y)

        COO_i = COO_i.long()
        COO_j = COO_j.long()
        valid_mask = (COO_i >= 0) & (COO_j >= 0)

        COO_i = COO_i[valid_mask]
        COO_j = COO_j[valid_mask]
        COO_values = COO_values[valid_mask]

        batch_i = COO_i // N
        index_i = COO_i % N
        batch_j = COO_j // M
        index_j = COO_j % M

        x_i = x[batch_i, index_i]
        y_j = y[batch_j, index_j]
        diff = x_i - y_j
        dist = torch.norm(diff, dim=1, keepdim=True).clamp(min=1e-6)

        grad_weighted = COO_values.unsqueeze(1) * (diff / dist)

        flat_x_idx = batch_i * N + index_i
        flat_y_idx = batch_j * M + index_j

        grad_x = grad_x.reshape(B * N, D)
        grad_y = grad_y.reshape(B * M, D)

        grad_x.index_add_(0, flat_x_idx, grad_weighted)
        grad_y.index_add_(0, flat_y_idx, -grad_weighted)

        grad_x = grad_x.reshape(B, N, D)
        grad_y = grad_y.reshape(B, M, D)

        grad_x *= grad_output
        grad_y *= grad_output

        return grad_x, grad_y, None, None


class APMLSparse(torch.nn.Module):
    def __init__(self, p_min=0.8, threshold=1e-10):
        super().__init__()
        self.p_min = p_min
        self.threshold = threshold

    def forward(self, x, y):
        return APMLSparseFunction.apply(x, y, self.p_min, self.threshold)
