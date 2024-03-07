from typing import Dict, Tuple
from torch import Tensor

from .kalman_filter import KalmanStep


class EKFStep(KalmanStep):
    def _get_correction(self, mean: Tensor, H: Tensor) -> Tensor:
        raise NotImplementedError

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        if kwargs['outlier_threshold'] > 0:
            raise NotImplementedError("Outlier rejection is not yet supported for EKF")

        orig_H = kwargs['H']
        correction = self._get_correction(mean, orig_H)
        newH = orig_H - correction

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs={'H': newH}
        )
