from typing import Tuple, Sequence

from torch import Tensor

from torchcast.kalman_filter import GaussianStep


class CensoredGaussianStep(GaussianStep):

    def update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError('pass lower/upper to _update; otherwise lots of unavoidable code duplication')

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                H: Tensor,
                R: Tensor,
                lower: Tensor,
                upper: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("use lower/upper")
