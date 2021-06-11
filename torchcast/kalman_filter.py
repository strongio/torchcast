"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts, in the form of
:class:`torchcast.state_space.Predictions`, which can be used for training
(:func:`~torchcast.state_space.Predictions.log_prob()`), evaluation
(:func:`~torchcast.state_space.Predictions.to_dataframe()`) or visualization
(:func:`~torchcast.state_space.Predictions.plot()`).

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.

----------
"""

from typing import Sequence, Dict, List

from torchcast.covariance import Covariance
from torchcast.process import Process
from torchcast.state_space.base import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep

from typing import Tuple, Type, Optional

import torch
from torch import nn, Tensor
from typing_extensions import Final


class GaussianStep(StateSpaceStep):
    """
    Used internally by `KalmanFilter` to apply the kalman-filtering algorithm. Subclasses can implement additional
    logic such as outlier-rejection, censoring, etc.
    """
    use_stable_cov_update: Final[bool] = True

    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.MultivariateNormal

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        Q = kwargs['Q']
        mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        cov = F @ cov @ F.permute(0, 2, 1) + Q
        return mean, cov

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, H: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        K = self.kalman_gain(cov=cov, H=H, R=R)
        measured_mean = (H @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (K @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = self.covariance_update(cov=cov, K=K, H=H, R=R)
        return new_mean, new_cov

    def covariance_update(self, cov: Tensor, K: Tensor, H: Tensor, R: Tensor) -> Tensor:
        I = torch.eye(cov.shape[1], dtype=cov.dtype, device=cov.device).unsqueeze(0)
        ikh = I - K @ H
        if self.use_stable_cov_update:
            return ikh @ cov @ ikh.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)
        else:
            return ikh @ cov

    def kalman_gain(self, cov: Tensor, H: Tensor, R: Tensor) -> Tensor:
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
    """
    ss_step_cls = GaussianStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None):

        if process_covariance is None:
            process_covariance = Covariance.for_processes(processes, cov_type='process')
        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)

        initial_covariance = Covariance.for_processes(processes, cov_type='initial')

        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance
        )

    def _build_var_mats(self,
                        static_kwargs: Dict[str, Dict[str, Tensor]],
                        time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                        num_groups: int,
                        out_timesteps: int) -> Tuple[List[Tensor], List[Tensor]]:

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

        # measure-variance:
        if 'measure_covariance' in time_varying_kwargs and \
                self.measure_covariance.expected_kwarg in time_varying_kwargs['measure_covariance']:
            mvar_inputs = time_varying_kwargs['measure_covariance'][self.measure_covariance.expected_kwarg]
            Rs: List[Tensor] = []
            for t in range(out_timesteps):
                Rs.append(self.measure_covariance(mvar_inputs[t]))
        else:
            mvar_input: Optional[Tensor] = None
            if 'measure_covariance' in static_kwargs and \
                    self.measure_covariance.expected_kwarg in static_kwargs['measure_covariance']:
                mvar_input = static_kwargs['measure_covariance'].get(self.measure_covariance.expected_kwarg)
            R = self.measure_covariance(mvar_input)
            if len(R.shape) == 2:
                R = R.expand(num_groups, -1, -1)
            Rs = [R] * out_timesteps

        return Rs, Qs
