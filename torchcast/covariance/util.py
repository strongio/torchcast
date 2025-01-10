from typing import Collection

import torch
from torch import Tensor


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def cov2corr(cov: Tensor) -> Tensor:
    std_ = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return cov / (std_.unsqueeze(-1) @ std_.unsqueeze(-2))


def mini_cov_mask(rank: int, empty_idx: Collection[int], **kwargs) -> Tensor:
    param_rank = rank - len(empty_idx)
    mask = torch.zeros((rank, param_rank), **kwargs)
    c = 0
    for r in range(rank):
        if r not in empty_idx:
            mask[r, c] = 1.
            c += 1
    return mask
