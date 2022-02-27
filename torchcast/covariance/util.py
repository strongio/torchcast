import torch
from torch import Tensor


def num_off_diag(rank: int) -> int:
    return int(rank * (rank - 1) / 2)


def cov2corr(cov: Tensor) -> Tensor:
    std_ = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
    return cov / (std_.unsqueeze(-1) @ std_.unsqueeze(-2))
