from typing import Tuple, Optional, Union, Dict

import torch

from torch import nn, Tensor

from .base import Process
from .utils import SingleOutput, Bounded


class LocalLevel(Process):
    """
    A process representing a random-walk.

    :param id: A unique identifier for this process.
    :param decay: If the process has decay, then the random walk will tend towards zero as we forecast out further
     (note that this means you should center your time-series, or you should include another process that does not
     have this decaying behavior). Decay can be between 0 and 1, but values < .50 (or even .90) can often be too
     rapid and you will run into trouble with vanishing gradients. When passing a pair of floats, the nn.Module will
     assign a parameter representing the decay as a learned parameter somewhere between these bounds.
    """

    def __init__(self,
                 id: str,
                 measure: Optional[str] = None,
                 decay: Optional[Union[torch.nn.Module, Tuple[float, float]]] = None):

        se = 'position'

        super(LocalLevel, self).__init__(
            id=id,
            measure=measure,
            state_elements=[se],
        )

        if decay:
            if isinstance(decay, bool):
                decay = (0.95, 1.00)
            if isinstance(decay, tuple):
                decay = SingleOutput(transform=Bounded(*decay))
            self.f_modules[f'{se}->{se}'] = decay
        else:
            self.f_tensors = {f'{se}->{se}': torch.ones(1)}

    def _build_h_mat(self, inputs: Dict[str, torch.Tensor], num_groups: int, num_times: int) -> Tensor:
        return torch.tensor([1.])


class LocalTrend(Process):
    """
    A process representing an evolving trend.

    :param id: A unique identifier for this process.
    :param decay_velocity: If set, then the trend will decay to zero as we forecast out further. The default is
     to allow the trend to decay somewhere between .95 (moderate decay) and 1.00 (no decay), with the exact value
     being a learned parameter.
    :param decay_position: See `decay` in :class:`LocalLevel`. Default is no decay.
    :param velocity_multi: Default 0.1. A multiplier on the velocity, so that
     ``next_position = position + velocity_multi * velocity``. A value of << 1.0 can be helpful since the
     trend has such a large effect on the prediction, so that large values can lead to exploding predictions.
    """

    def __init__(self,
                 id: str,
                 measure: Optional[str] = None,
                 decay_velocity: Optional[Union[torch.nn.Module, Tuple[float, float]]] = (.95, 1.00),
                 decay_position: Optional[Union[torch.nn.Module, Tuple[float, float]]] = None,
                 velocity_multi: float = 0.1):

        # define transitions:
        self.f_modules: nn.ModuleDict = nn.ModuleDict()
        self.f_tensors: Dict[str, torch.Tensor] = {}

        if decay_position is None:
            self.f_tensors['position->position'] = torch.ones(1)
        else:
            if isinstance(decay_position, tuple):
                decay_position = SingleOutput(transform=Bounded(*decay_position))
            self.f_modules['position->position'] = decay_position
        if decay_velocity is None:
            self.f_tensors['velocity->velocity'] = torch.ones(1)
        else:
            if isinstance(decay_velocity, tuple):
                decay_velocity = SingleOutput(transform=Bounded(*decay_velocity))
            self.f_modules['velocity->velocity'] = decay_velocity

        assert velocity_multi <= 1.
        self.f_tensors['velocity->position'] = torch.ones(1) * velocity_multi

        super(LocalTrend, self).__init__(
            id=id,
            measure=measure,
            state_elements=['position', 'velocity'],
        )

    def _build_h_mat(self, inputs: Dict[str, torch.Tensor], num_groups: int, num_times: int) -> Tensor:
        return torch.tensor([1., 0.])
