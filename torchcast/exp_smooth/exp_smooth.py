"""
The :class:`.ExpSmoother` is a :class:`torch.nn.Module` which generates forecasts using exponential smoothing.

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.
"""
from typing import Sequence, Optional, Tuple, List, Dict, Iterable

import torch
from torch import Tensor

from torchcast.exp_smooth.smoothing_matrix import SmoothingMatrix
from torchcast.covariance import Covariance
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel, Predictions
from torchcast.state_space.ss_step import StateSpaceStep


class ExpSmoothStep(StateSpaceStep):

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        new_kwargs = kwargs.copy()
        if val_idx is None:
            for k in ['H', 'R', 'K']:
                new_kwargs[k] = kwargs[k][groups]
            return input[groups], new_kwargs
        else:
            m1d = torch.meshgrid(groups, val_idx, indexing='ij')
            m2d = torch.meshgrid(groups, val_idx, val_idx, indexing='ij')
            masked_input = input[m1d[0], m1d[1]]
            new_kwargs.update({
                'H': kwargs['H'][m1d[0], m1d[1]],
                'R': kwargs['R'][m2d[0], m2d[1], m2d[2]],
                'K': kwargs['K'][m1d[0], m1d[1]],
            })
            return masked_input, new_kwargs

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Optional[Tensor]]:
        measured_mean = (kwargs['H'] @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (kwargs['K'] @ resid.unsqueeze(-1)).squeeze(-1)
        return new_mean, None

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        new_mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        new_cov = kwargs['cov1step']
        if cov is not None:
            new_cov = new_cov + F @ cov @ F.permute(0, 2, 1)
        return new_mean, new_cov


class ExpSmoother(StateSpaceModel):
    """
    Uses exponential smoothing to generate forecasts.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    """
    ss_step_cls = ExpSmoothStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 measure_covariance: Optional[Covariance] = None,
                 smoothing_matrix: Optional[SmoothingMatrix] = None):

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
        )

        if smoothing_matrix is None:
            smoothing_matrix = SmoothingMatrix.from_measures_and_processes(measures=measures, processes=processes)
        self.smoothing_matrix = smoothing_matrix.set_id('smoothing_matrix')

    @torch.jit.ignore
    def initial_covariance(self, inputs: dict, num_groups: int, num_times: int, _ignore_input: bool = False) -> Tensor:
        # this is a dummy method since we just need the shape to be like what you'd get from calling Covariance()(),
        # but don't actually want learnable parameter b/c unused otherwise
        ms = self._get_measure_scaling()
        return torch.eye(self.state_rank, dtype=ms.dtype, device=ms.device).expand(num_groups, num_times, -1, -1)

    @torch.jit.ignore
    def design_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'measure_covariance', self.measure_covariance
        yield 'smoothing_matrix', self.smoothing_matrix

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        assert out_timesteps

        Fs, Hs = self._build_transition_and_measure_mats(kwargs_per_process, num_groups, out_timesteps)

        # measure-variance:
        mcov_input = kwargs_per_process.get('measure_covariance', {})
        Rs = self.measure_covariance(mcov_input,
                                     num_groups=num_groups,
                                     num_times=out_timesteps)

        sm_input = kwargs_per_process.get('smoothing_matrix', {})
        Ks = self.smoothing_matrix(sm_input,
                                   num_groups=num_groups,
                                   num_times=out_timesteps)

        # pre-compute 1-step-ahead variance
        cov1steps: Optional[List[Tensor]] = None
        if len(mcov_input) > 0 or len(sm_input) > 0:
            cov1steps = (Ks @ Rs @ Ks.transpose(-1, -2)).unbind(1)

        # unbind
        Rs = Rs.unbind(1)
        Ks = Ks.unbind(1)
        if cov1steps is None:
            # if not time-varying, the this is cheaper
            # note: will unnecessarily do the less cheap version with inputs that are groupwise but not timewise
            cov1step = Ks[0] @ Rs[0] @ Ks[0].transpose(-1, -2)
            cov1steps = [cov1step] * out_timesteps

        predict_kwargs = {'F': Fs, 'cov1step': cov1steps}
        update_kwargs = {'H': Hs, 'K': Ks, 'R': Rs}
        return predict_kwargs, update_kwargs

    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'Predictions':
        del kwargs['K']
        return super()._generate_predictions(
            preds=preds,
            updates=updates,
            **kwargs
        )
