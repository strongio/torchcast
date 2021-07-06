from typing import Dict, List

import torch
from torch import Tensor


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def pad_covariance(unpadded_cov: Tensor, mask_1d: List[int]) -> Tensor:
    rank = len(mask_1d)
    padded_to_unpadded: Dict[int, int] = {}
    up_idx = 0
    for p_idx, is_filled in enumerate(mask_1d):
        if is_filled == 1:
            padded_to_unpadded[p_idx] = up_idx
            up_idx += 1
    if up_idx == len(mask_1d):
        # shortcut
        return unpadded_cov

    out = torch.zeros(unpadded_cov.shape[:-2] + (rank, rank), device=unpadded_cov.device, dtype=unpadded_cov.dtype)
    for to_r in range(rank):
        for to_c in range(to_r, rank):
            from_r = padded_to_unpadded.get(to_r)
            from_c = padded_to_unpadded.get(to_c)
            if from_r is not None and from_c is not None:
                out[..., to_r, to_c] = unpadded_cov[..., from_r, from_c]
                if to_r != to_c:
                    out[..., to_c, to_r] = out[..., to_r, to_c]  # symmetrical
    return out


def cov2corr(cov: Tensor) -> Tensor:
    std_ = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return cov / (std_.unsqueeze(-1) @ std_.unsqueeze(-2))
