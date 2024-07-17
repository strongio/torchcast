from typing import Dict, Tuple, Optional
from torch import Tensor

from .kalman_filter import KalmanStep


class EKFStep(KalmanStep):
    def _adjust_h(self, mean: Tensor, H: Tensor) -> Tensor:
        return H

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor]) -> Tensor:
        assert R is not None
        return R

    def _link(self, measured_mean: Tensor) -> Tensor:
        return measured_mean

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        if (kwargs['outlier_threshold'] != 0).any():
            raise NotImplementedError("Outlier rejection is not yet supported for EKF")

        orig_H = kwargs['H']
        h_dot_state = (orig_H @ mean.unsqueeze(-1)).squeeze(-1)
        kwargs['measured_mean'] = self._link(h_dot_state)
        kwargs['H'] = self._adjust_h(mean, orig_H)
        kwargs['R'] = self._adjust_r(kwargs['measured_mean'], kwargs.get('R', None))

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs=kwargs
        )
