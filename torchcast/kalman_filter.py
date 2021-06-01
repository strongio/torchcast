"""
The :class:`.KalmanFilter` is a :class:`torch.nn.Module` which generates forecasts, in the form of
:class:`torchcast.state_space.Predictions`, which can be used for training
(:func:`~torchcast.state_space.Predictions.log_prob()`), evaluation
(:func:`~torchcast.state_space.Predictions.to_dataframe()`) or visualization
(:func:`~torchcast.state_space.Predictions.plot()`).

This class inherits most of its methods from :class:`torchcast.state_space.StateSpaceModel`.

----------
"""

from typing import Sequence

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
                 measure_covariance: Optional[Covariance] = None,
                 initial_covariance: Optional[Covariance] = None):

        if process_covariance is None:
            process_covariance = Covariance.for_processes(processes, cov_type='process')
        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)
        if initial_covariance is None:
            initial_covariance = Covariance.for_processes(processes, cov_type='initial')
        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance
        )
