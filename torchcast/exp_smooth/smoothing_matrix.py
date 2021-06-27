import math
from typing import Iterable, Tuple, Optional, Sequence, List
from warnings import warn

import torch
from torch import jit, Tensor

from torchcast.process import Process
from torchcast.internals.utils import get_owned_kwarg


class SmoothingMatrix(torch.nn.Module):

    @classmethod
    def from_measures_and_processes(cls,
                                    measures: Sequence[str],
                                    processes: Sequence['Process'],
                                    method: str = 'full',
                                    predict_module: Optional[torch.nn.Module] = None,
                                    **kwargs) -> 'SmoothingMatrix':
        """

        :param measures:
        :param processes:
        :param method:
        :param predict_module:
        :param kwargs:
        :return:
        """
        measures = list(measures)
        state_rank = 0
        fixed_idx = []
        for p in processes:
            midx = measures.index(p.measure)
            fixed_els = p.fixed_state_elements or []
            for i, se in enumerate(p.state_elements):
                if se in fixed_els:
                    fixed_idx.append((state_rank + i, midx))
                # TODO: warn if time-varying H but some unfixed_els
            state_rank += len(p.state_elements)
        return cls(
            measure_rank=len(measures),
            state_rank=state_rank,
            empty_idx=fixed_idx,
            method=method,
            predict_module=predict_module,
            **kwargs
        )

    def __init__(self,
                 measure_rank: int,
                 state_rank: int,
                 method: str = 'full',
                 empty_idx: List[Tuple[int, int]] = (),
                 init_bias: int = -10,
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

        self.init_bias = init_bias

        self.unconstrained_params: Optional[torch.nn.Parameter] = None
        self.lr1: Optional[torch.nn.Parameter] = None
        self.lr2: Optional[torch.nn.Parameter] = None
        if method == 'full':
            self.method = method
            self.unconstrained_params = torch.nn.Parameter(.1 * torch.randn(len(self.full_idx)))
        elif method.startswith('low_rank'):
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                low_rank = int(math.sqrt((self.state_rank * self.measure_rank) / (self.state_rank + self.measure_rank)))
            self.lr1 = torch.nn.Parameter(.1 * torch.randn(self.state_rank, low_rank))
            self.lr2 = torch.nn.Parameter(.1 * torch.randn(low_rank, self.measure_rank))
        else:
            raise ValueError(f"Unrecognized method `{method}`")

        self.predict_module = predict_module

        # TODO: allow user-defined?
        self.expected_kwarg = '' if self.predict_module is None else 'X'

    @property
    def param_rank(self) -> int:
        return len(self.unconstrained_params)

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        if self.predict_module is not None and not _ignore_input:
            raise NotImplementedError

        if self.method == 'full':
            out = torch.zeros(
                (self.state_rank, self.measure_rank),
                dtype=self.unconstrained_params.dtype,
                device=self.unconstrained_params.device
            )
            for i, (r, c) in enumerate(self.full_idx):
                out[..., r, c] = torch.sigmoid(self.unconstrained_params[i] + self.init_bias)
        elif self.method == 'low_rank':
            out = torch.sigmoid(self.lr1 @ self.lr2 + self.init_bias)
            # TODO: this could be more efficient
            mask = torch.zeros_like(out)
            for i, (r, c) in enumerate(self.full_idx):
                mask[..., r, c] = 1.
            out = out * mask
        else:
            raise RuntimeError

        return out

    @jit.ignore
    def set_id(self, id: str) -> 'SmoothingMatrix':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, Tensor]]:
        if self.expected_kwarg:
            found_key, value = get_owned_kwarg(self.id, self.expected_kwarg, kwargs)
            yield found_key, self.expected_kwarg, torch.as_tensor(value)
