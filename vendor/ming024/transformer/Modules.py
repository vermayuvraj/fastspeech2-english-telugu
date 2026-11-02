import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        # Same scaled attention, using PyTorch's memory-efficient CUDA kernels.
        allowed = (~mask).unsqueeze(1) if mask is not None else None
        output = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), attn_mask=allowed
        ).squeeze(1)
        return output, None
