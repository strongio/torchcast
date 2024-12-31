from typing import Tuple, Sequence, List, Dict, Optional, Callable

import torch

from torch import nn, Tensor, jit


class Process(nn.Module):
    """
    This is the base class. The process is defined by the state-elements it generates predictions for. It generates
    two kinds of predictions: (1) the observation matrix, (2) the transition matrix.

    :param id: Unique identifier for the process
    :param state_elements: List of strings with the state-element names
    :param measure: The name of the measure for this process.
    :param fixed_state_elements: Names of ``state_elements`` that are 'fixed'. In a kalman-filter these will be
     initially responsive to the incoming data but gradually convergee over time; in an exponential-smoothing model
     these will be fixed at their initial value.
    """

    def __init__(self,
                 id: str,
                 state_elements: Sequence[str],
                 measure: Optional[str] = None,
                 fixed_state_elements: Optional[List[str]] = None):

        super(Process, self).__init__()
        self.id = id

        self._info: Tensor
        self.register_buffer('_info', torch.empty(0), persistent=False)  # for dtype/device info

        # state elements:
        self.state_elements = state_elements
        self.se_to_idx = {se: i for i, se in enumerate(self.state_elements)}
        assert len(state_elements) == len(self.se_to_idx), f"state-elements are not unique:{state_elements}"

        # can be populated later, as long as it's before torch.jit.script
        self.measure: str = '' if measure is None else measure

        # elements without process covariance, defaults to none
        if fixed_state_elements is not None:
            extra = set(fixed_state_elements) - set(state_elements)
            if extra:
                raise ValueError(f"fixed_state_elements contains elements not in state_elements: {extra}")
        self.fixed_state_elements: Optional[List[str]] = fixed_state_elements

        # can/should be overridden by subclasses:
        self.expected_kwargs: Optional[List[str]] = None
        self.f_modules: nn.ModuleDict = nn.ModuleDict()
        self.f_tensors: Dict[str, torch.Tensor] = {'': torch.empty(0)}  # for jit

    @jit.ignore
    def offset_initial_state(self, initial_state: Tensor, start_offsets: Optional[Sequence] = None) -> Tensor:
        return initial_state

    def forward(self, inputs: Dict[str, Tensor], num_groups: int, num_times: int) -> Tuple[Tensor, Tensor]:
        """
        :param inputs: Inputs from ``forward()``
        :param num_groups: Number of groups.
        :param num_times: Number of timesteps.
        :return: A tuple of tensors: the observation matrices (H) and the transition matrices (F). Each has batch-dims
        ``(num_groups, num_times)``.
        """
        H = self._build_h_mat(inputs, num_groups, num_times)
        F = self._build_f_mat(inputs, num_groups, num_times)
        return H, F

    def _build_h_mat(self, inputs: Dict[str, Tensor], num_groups: int, num_times: int) -> Tensor:
        """
        Construct observation matrix H.

        :param inputs: Inputs from ``forward()``
        :param num_groups: Number of groups.
        :param num_times: Number of timesteps.
        :return: A tensor of shape ``(num_groups, num_times, num_states)``
        """
        raise NotImplementedError

    def _get_transitions_dict(self, inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Construct transitions dictionary which will be used by ``_build_f_mat()`` to construct transition-matrix F.
        This default method does not make use of inputs, but subclasses can override (e.g. different decay for each
        group).

        :param inputs: Inputs from ``forward()``
        :return: A dictionary of tensors, where keys indicate ``{from}->{to}`` transitions, and values are the
        transition-values.
        """
        out = {}
        for km, module in self.f_modules.items():
            out[km] = module()
        for kt, tens in self.f_tensors.items():
            if kt == '':  # see self.f_tensors definition
                continue
            # can't register f_tensors as buffers, see https://github.com/pytorch/pytorch/issues/43815
            # another approach would be override _apply, but that doesn't seem to work with torchscript
            out[kt] = tens.to(dtype=self.dtype, device=self.device)
        return out

    def _build_f_mat(self, inputs: Dict[str, Tensor], num_groups: int, num_times: int) -> Tensor:
        """
        :param inputs: Inputs from ``forward()``
        :param num_groups: Number of groups.
        :param num_times: Number of timesteps.
        :return:  A tensor of shape ``(num_groups, num_times, num_states, num_states)``
        """
        transitions = self._get_transitions_dict(inputs)
        F = torch.zeros(num_groups, num_times, self.rank, self.rank, dtype=self.dtype, device=self.device)
        for from__to, tens in transitions.items():
            assert tens is not None
            r, c = self._transition_key_to_rc(from__to)
            F[:, :, r, c] = tens

        if torch.isnan(F).any() or torch.isinf(F).any():
            raise RuntimeError(f"{self.id} produced F with nans")
        return F

    @property
    def rank(self) -> int:
        return len(self.state_elements)

    def _transition_key_to_rc(self, transition_key: str) -> Tuple[List[int], List[int]]:
        from_el, sep, to_el = transition_key.partition("->")
        if sep == '':
            assert from_el == 'all_self', f"Expected '[from_el]->[to_el]', or 'all_self'. Got '{transition_key}'"
            return list(range(self.rank)), list(range(self.rank))
        else:
            c = self.se_to_idx[from_el]
            r = self.se_to_idx[to_el]
            return [r], [c]

    @property
    def device(self) -> torch.device:
        return self._info.device

    @property
    def dtype(self) -> torch.dtype:
        return self._info.dtype

    def __repr__(self) -> str:
        return f'{type(self).__name__}(id={repr(self.id)})'
