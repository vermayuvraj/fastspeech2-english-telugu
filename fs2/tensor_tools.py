"""Minimal device-safe replacements for the reference plotting utility imports."""
import torch
import torch.nn.functional as F


def get_mask_from_lengths(lengths, max_len=None):
    max_len = int(lengths.max().item()) if max_len is None else int(max_len)
    return torch.arange(max_len, device=lengths.device)[None, :] >= lengths[:, None]


def pad(tensors, max_len=None):
    max_len = max(t.shape[0] for t in tensors) if max_len is None else int(max_len)
    return torch.stack([F.pad(t, (0, 0, 0, max_len - t.shape[0])) for t in tensors])
