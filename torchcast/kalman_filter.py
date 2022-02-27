"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts using the full kalman-filtering
algorithm.

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.

----------
"""

from typing import Sequence, Dict, List, Iterable

from torchcast.state_space import Predictions
from torchcast.covariance import Covariance
from torchcast.process import Process
from torchcast.state_space.base import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep

from typing import Tuple, Optional

import torch
from torch import nn, Tensor
from typing_extensions import Final


class KalmanStep(StateSpaceStep):
    """
    Used internally by ``KalmanFilter`` to apply the kalman-filtering algorithm. Subclasses can implement additional
    logic such as outlier-rejection, censoring, etc.
    """
    use_stable_cov_update: Final[bool] = True

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        Q = kwargs['Q']
        mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        cov = F @ cov @ F.permute(0, 2, 1) + Q
        return mean, cov

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        if val_idx is None:
            return input[groups], {k: v[groups] for k, v in kwargs.items()}
        else:
            m1d = torch.meshgrid(groups, val_idx, indexing='ij')
            m2d = torch.meshgrid(groups, val_idx, val_idx, indexing='ij')
            masked_input = input[m1d[0], m1d[1]]
            masked_kwargs = {
                'H': kwargs['H'][m1d[0], m1d[1]],
                'R': kwargs['R'][m2d[0], m2d[1], m2d[2]]
            }
            return masked_input, masked_kwargs

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        H = kwargs['H']
        R = kwargs['R']
        K = self._kalman_gain(cov=cov, H=H, R=R)
        measured_mean = (H @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (K @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = self._covariance_update(cov=cov, K=K, H=H, R=R)
        return new_mean, new_cov

    def _covariance_update(self, cov: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        ikh = I - K @ H
        if self.use_stable_cov_update:
            return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)
        else:
            return ikh @ cov

    @staticmethod
    def _kalman_gain(cov: Tensor, H: Tensor, R: Tensor) -> Tensor:
        Ht = H.permute(0, 2, 1)
        covs_measured = cov @ Ht
        system_covariance = H @ cov @ Ht + R
        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt = torch.linalg.solve(A, B)
        K = Kt.permute(0, 2, 1)
        return K


class KalmanFilter(StateSpaceModel):
    """
    Uses the full kalman-filtering algorithm for generating forecasts.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param process_covariance: A module created with ``Covariance.from_processes(processes)``.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    """
    ss_step_cls = KalmanStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None):

        initial_covariance = Covariance.from_processes(processes, cov_type='initial')

        if process_covariance is None:
            process_covariance = Covariance.from_processes(processes, cov_type='process')

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
        )
        self.process_covariance = process_covariance.set_id('process_covariance')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')

    @torch.jit.ignore()
    def design_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'process_covariance', self.process_covariance
        yield 'measure_covariance', self.measure_covariance
        yield 'initial_covariance', self.initial_covariance

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        Fs, Hs = self._build_transition_and_measure_mats(kwargs_per_process, num_groups, out_timesteps)

        # measure-variance:
        Rs = self.measure_covariance(kwargs_per_process.get('measure_covariance', {}),
                                     num_groups=num_groups,
                                     num_times=out_timesteps)
        Rs = Rs.unbind(1)

        # process-variance:
        measure_scaling = torch.diag_embed(self._get_measure_scaling().unsqueeze(0).unsqueeze(0))
        pcov_raw = self.process_covariance(
            kwargs_per_process.get('process_covariance', {}), num_groups=num_groups, num_times=out_timesteps
        )
        Qs = measure_scaling @ pcov_raw @ measure_scaling
        Qs = Qs.unbind(1)

        predict_kwargs = {'F': Fs, 'Q': Qs}
        update_kwargs = {'H': Hs, 'R': Rs}
        return predict_kwargs, update_kwargs
