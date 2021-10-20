from warnings import warn
from typing import Optional, Iterable, Tuple

import torch
from torch import Tensor, jit


class SumCovariance(torch.nn.Module):
    def __init__(self, *covs):
        super(SumCovariance, self).__init__()
        self.covs = torch.nn.ModuleList(covs)
        self.id = None

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        stacked = torch.stack([cov(input=input, _ignore_input=_ignore_input) for cov in self.covs], 0)
        return torch.sum(stacked, 0)

    @jit.ignore
    def set_id(self, id: str):
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self
