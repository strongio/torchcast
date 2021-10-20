from itertools import zip_longest
from typing import Sequence, Optional, Iterable, Tuple
from warnings import warn

import torch
from torch import Tensor, jit

from torchcast.covariance import Covariance
from torchcast.process import Process


class BlockDiagCovariance(torch.nn.Module):
    @classmethod
    def from_measures_and_processes(cls,
                                    measures: Sequence[str],
                                    processes: Sequence['Process'],
                                    cov_type: str = 'process') -> 'BlockDiagCovariance':
        """
        :param processes: The ``processes`` used in your :class:`.StateSpaceModel`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :return: A :class:`.BlockDiagCovariance` object that can be used in your :class:`.KalmanFilter`.
        """
        _found_measures = []
        block_sizes = {m: 0 for m in measures}
        empty_idxs = {m: [] for m in measures}
        for p in processes:
            if p.measure not in _found_measures:
                _found_measures.append(p.measure)

            # state-rank:
            block_sizes[p.measure] += len(p.state_elements)

            # no proc-var:
            no_cov_elements = []
            if cov_type == 'process':
                no_cov_elements = p.fixed_state_elements or []
            for i, se in enumerate(p.state_elements):
                if se in no_cov_elements:
                    empty_idxs[p.measure].append(i)
        if _found_measures != list(measures) and set(_found_measures) == set(measures):
            raise NotImplementedError(
                "``BlockDiagCovariance`` currently only works if processes are in same order as measures"
            )

        return cls(block_sizes=[block_sizes[m] for m in measures], empty_idxs=[empty_idxs[m] for m in measures])

    def __init__(self,
                 block_sizes: Sequence[int],
                 empty_idxs: Sequence[Sequence[int]],
                 block_method: str = 'log_cholesky'):
        super(BlockDiagCovariance, self).__init__()
        self.blocks = torch.nn.ModuleList([
            Covariance(rank=bs, empty_idx=ei, method=block_method)
            for bs, ei in zip_longest(block_sizes, empty_idxs)
        ])
        self.id = None

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        blocks = [block(input=input, _ignore_input=_ignore_input) for block in self.blocks]
        rank = sum(block.shape[-1] for block in blocks)
        out = torch.zeros((rank, rank), dtype=blocks[0].dtype, device=blocks[0].device)
        start_ = 0
        for block in blocks:
            end_ = start_ + block.shape[-1]
            out[..., start_:end_, start_:end_] = block
            start_ = end_
        return out

    @jit.ignore
    def set_id(self, id: str):
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self
