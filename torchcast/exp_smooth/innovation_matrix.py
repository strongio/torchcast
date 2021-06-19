from typing import Iterable, Tuple, Optional, Sequence, List
from warnings import warn

import torch
from torch import jit, Tensor

from torchcast.process import Process
from torchcast.internals.utils import get_owned_kwarg


class InnovationMatrix(torch.nn.Module):

    def __init__(self,
                 measure_rank: int,
                 state_rank: int,
                 empty_idx: List[Tuple[int, int]] = (),
                 predict_module: Optional[torch.nn.Module] = None,
                 id: Optional[str] = None):

        super().__init__()

        self.state_rank = state_rank
        self.measure_rank = measure_rank

        self.full_idx = []
        empty_idx = set(empty_idx)
        for r in range(self.state_rank):
            for c in range(self.measure_rank):
                if (r, c) in empty_idx:
                    continue
                self.full_idx.append((r, c))
        self.id = id
        self.unconstrained_params = torch.nn.Parameter(.1 * torch.randn(len(self.full_idx)))
        self.predict_module = predict_module

    @property
    def param_rank(self) -> int:
        return len(self.unconstrained_params)

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        if self.predict_module is not None and not _ignore_input:
            pred = self.predict_module(input)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise RuntimeError(f"{self.id}'s `predict_module` produced nans/infs")
            if len(pred.shape) == 1:
                raise ValueError(
                    f"{self.id} `predict_variance` module output should have 2D output, got {len(pred.shape)}"
                )
            elif pred.shape[-1] not in (1, self.param_rank):
                raise ValueError(
                    f"{self.id} `predict_module` module output should have `shape[-1]` of "
                    f"{self.param_rank}, got {pred.shape[-1]}"
                )
            pred = self.unconstrained_params + pred
        else:
            pred = self.unconstrained_params

        out = torch.zeros(
            pred.shape[0:1] + (self.state_rank, self.measure_rank),
            dtype=self.unconstrained_params.dtype,
            device=self.unconstrained_params.device
        )
        import pdb
        pdb.set_trace()
        out[:, self.full_idx] = pred
        return pred

    @jit.ignore
    def set_id(self, id: str) -> 'InnovationMatrix':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, Tensor]]:
        if self.expected_kwarg:
            found_key, value = get_owned_kwarg(self.id, self.expected_kwarg, kwargs)
            yield found_key, self.expected_kwarg, torch.as_tensor(value)
