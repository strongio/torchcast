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
            m1d = torch.meshgrid(groups, val_idx)
            m2d = torch.meshgrid(groups, val_idx, val_idx)
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
        system_covariance = torch.baddbmm(R, H @ cov, Ht)
        A = system_covariance.permute(0, 2, 1)
        B = covs_measured.permute(0, 2, 1)
        Kt, _ = torch.solve(B, A)
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
            R=torch.stack(update_kwargs['R'], 1),
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

        # process-variance:
        measure_scaling = torch.diag_embed(self._get_measure_scaling())
        if 'process_covariance' in time_varying_kwargs and \
                self.process_covariance.expected_kwarg in time_varying_kwargs['process_covariance']:
            pvar_inputs = time_varying_kwargs['process_covariance'][self.process_covariance.expected_kwarg]
            Qs: List[Tensor] = []
            for t in range(out_timesteps):
                Qs.append(measure_scaling @ self.process_covariance(pvar_inputs[t]) @ measure_scaling)
        else:
            pvar_input: Optional[Tensor] = None
            if 'process_covariance' in static_kwargs and \
                    self.process_covariance.expected_kwarg in static_kwargs['process_covariance']:
                pvar_input = static_kwargs['process_covariance'].get(self.process_covariance.expected_kwarg)
            Q = measure_scaling @ self.process_covariance(pvar_input) @ measure_scaling
            if len(Q.shape) == 2:
                Q = Q.expand(num_groups, -1, -1)
            Qs = [Q] * out_timesteps

        predict_kwargs = {'F': Fs, 'Q': Qs}
        update_kwargs = {'H': Hs, 'R': Rs}
        return predict_kwargs, update_kwargs
