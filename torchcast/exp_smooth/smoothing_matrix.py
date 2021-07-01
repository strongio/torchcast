import math
from typing import Iterable, Tuple, Optional, Sequence
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
        fixed_states = []
        for p in processes:
            fixed_els = p.fixed_state_elements or []
            for i, se in enumerate(p.state_elements):
                if se in fixed_els:
                    fixed_states.append(state_rank + i)
                # TODO: warn if time-varying H but some unfixed_els
            state_rank += len(p.state_elements)
        return cls(
            measure_rank=len(measures),
            state_rank=state_rank,
            fixed_states=fixed_states,
            method=method,
            predict_module=predict_module,
            **kwargs
        )

    def __init__(self,
                 measure_rank: int,
                 state_rank: int,
                 method: str = 'full',
                 fixed_states: Sequence[int] = (),
                 init_bias: int = -10,
                 predict_module: Optional[torch.nn.Module] = None,
                 id: Optional[str] = None):

        super().__init__()

        self.id = id

        self.state_rank = state_rank
        self.measure_rank = measure_rank
        self.full_states = [i for i in range(self.state_rank) if i not in fixed_states]

        self.init_bias = init_bias

        self.unconstrained_params: Optional[torch.nn.Parameter] = None
        self.lr1: Optional[torch.nn.Parameter] = None
        self.lr2: Optional[torch.nn.Parameter] = None
        if method == 'full':
            self.method = method
            self.unconstrained_params = torch.nn.Parameter(.1 * torch.randn(self.measure_rank * len(self.full_states)))
        elif method.startswith('low_rank'):
            if measure_rank == 1:
                warn("Using `method='low_rank'` with 1 measure")
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                ub = (len(self.full_states) * self.measure_rank) / (len(self.full_states) + self.measure_rank)
                low_rank = int(math.sqrt(ub))
            self.lr1 = torch.nn.Parameter(.1 * torch.randn(len(self.full_states), low_rank))
            self.lr2 = torch.nn.Parameter(.1 * torch.randn(low_rank, self.measure_rank))
        else:
            raise ValueError(f"Unrecognized method `{method}`")

        self.predict_module = predict_module

        # TODO: allow user-defined?
        self.expected_kwarg = '' if self.predict_module is None else 'X'

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        if self.predict_module is not None and not _ignore_input:
            # TODO: `multi_forward()` that only computes lr1 @ lr2 once if we have time-varying inputs
            raise NotImplementedError

        if self.method == 'full':
            K = torch.zeros(
                (self.state_rank, self.measure_rank),
                dtype=self.unconstrained_params.dtype,
                device=self.unconstrained_params.device
            )
            k_full = torch.sigmoid(self.unconstrained_params + self.init_bias)
            K[..., self.full_states, :] = k_full.view(len(self.full_states), self.measure_rank)
            return K
        elif self.method == 'low_rank':
            sm = torch.nn.Softmax(-1)
            # we want lr1 @ lr2 to be constrained to 0-1. this could be accomplished by applying sigmoid after matmul,
            # but there may be matmul-ordering optimizations to keeping lr1 and lr2 separate for `predict` or `update`
            # so we want to instead constrain lr1 and lr2 themselves
            # lr1: each row needs to sum to 0<=sum(row)<=1. softmax is traditionally overparameterized b/c sums to 1,
            #      but here it is not overparameterized b/c sums to <=1 (`* torch.sigmoid(l1.sum)` part)
            lr1 = torch.zeros((self.state_rank, self.lr1.shape[-1]), dtype=self.lr1.dtype, device=self.lr1.device)
            lr1[self.full_states] = sm(self.lr1) * torch.sigmoid(self.lr1.sum(-1, keepdim=True) + self.init_bias / 2)
            # lr2: each element is 0<=el<=1
            lr2 = torch.sigmoid(self.lr2 + self.init_bias / 2)
            # currently not outputting lr1,lr2 separately, as overhead of of extra matmul calls is worse than reduced
            # theoretical number of ops. but may re-examine in the future.
            return lr1 @ lr2
        else:
            raise RuntimeError

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
