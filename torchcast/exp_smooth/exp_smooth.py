"""
The :class:`.ExpSmoother` is a :class:`torch.nn.Module` which generates forecasts using exponential smoothing.

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.

----------
"""
from typing import Sequence, Optional, Tuple, List, Dict, Iterable

import torch
from torch import Tensor

from torchcast.state_space import Predictions
from torchcast.exp_smooth.smoothing_matrix import SmoothingMatrix
from torchcast.covariance import Covariance
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep


class ExpSmoothStep(StateSpaceStep):

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        if val_idx is None:
            return input[groups], {k: v[groups] for k, v in kwargs.items()}
        else:
            m1d = torch.meshgrid(groups, val_idx)
            m2d = torch.meshgrid(groups, val_idx, val_idx)
            masked_input = input[m1d[0], m1d[1]]
            masked_kwargs = {
                'H': kwargs['H'][m1d[0], m1d[1]],
                'R': kwargs['R'][m2d[0], m2d[1], m2d[2]],
                'K': kwargs['K'][m1d[0], m1d[1]],
            }
            return masked_input, masked_kwargs

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        measured_mean = (kwargs['H'] @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (kwargs['K'] @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = torch.zeros_like(cov)
        return new_mean, new_cov

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        K = kwargs['K']
        R = kwargs['R']
        new_mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        new_cov = K @ R @ K.permute(0, 2, 1)
        if (cov != 0).any():
            new_cov = new_cov + F @ cov @ F.permute(0, 2, 1)
        return new_mean, new_cov


class ExpSmoother(StateSpaceModel):
    """
    Uses exponential smoothing to generate forecasts.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    :param predict_smoothing: A ``torch.nn.Module`` which predicts the smoothing parameters. The module should predict
     these as real-values and they will be constrained to 0-1 internally.
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
    def initial_covariance(self, *args, **kwargs) -> Tensor:
        ms = self._get_measure_scaling()
        return torch.eye(self.state_rank, dtype=ms.dtype, device=ms.device)

    @torch.jit.ignore
    def design_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'measure_covariance', self.measure_covariance
        yield 'smoothing_matrix', self.smoothing_matrix

    @torch.jit.ignore
    def _generate_predictions(self,
                              means: Tensor,
                              covs: Tensor,
                              predict_kwargs: Dict[str, List[Tensor]],
                              update_kwargs: Dict[str, List[Tensor]]) -> 'Predictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """
        return Predictions(
            state_means=means,
            state_covs=covs,
            R=torch.stack(predict_kwargs['R'], 1),
            H=torch.stack(update_kwargs['H'], 1),
            kalman_filter=self
        )

    def _build_design_mats(self,
                           static_kwargs: Dict[str, Dict[str, Tensor]],
                           time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        Fs, Hs = self._build_transition_and_measure_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)

        # measure-variance:
        Rs = self._build_measure_var_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)

        # innovation matrix / kalman-gain:
        if 'smoothing_matrix' in time_varying_kwargs and \
                self.smoothing_matrix.expected_kwarg in time_varying_kwargs['smoothing_matrix']:
            pvar_inputs = time_varying_kwargs['smoothing_matrix'][self.smoothing_matrix.expected_kwarg]
            Ks: List[Tensor] = []
            for t in range(out_timesteps):
                Ks.append(self.smoothing_matrix(pvar_inputs[t]))
        else:
            pvar_input: Optional[Tensor] = None
            if 'smoothing_matrix' in static_kwargs and \
                    self.smoothing_matrix.expected_kwarg in static_kwargs['smoothing_matrix']:
                pvar_input = static_kwargs['smoothing_matrix'].get(self.smoothing_matrix.expected_kwarg)
            K = self.smoothing_matrix(pvar_input)
            if len(K.shape) == 2:
                K = K.expand(num_groups, -1, -1)
            Ks = [K] * out_timesteps

        predict_kwargs = {'F': Fs, 'K': Ks, 'R': Rs}
        update_kwargs = {'H': Hs, 'K': Ks}
        return predict_kwargs, update_kwargs
