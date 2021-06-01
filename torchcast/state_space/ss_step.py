from typing import Type, Tuple, Dict

import torch
from torch import Tensor

from torchcast.internals.utils import get_nan_groups


class StateSpaceStep(torch.nn.Module):
    """
    Used internally by `StateSpaceModel` to apply the predict/update steps.
    """

    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        raise NotImplementedError

    def forward(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                predict_kwargs: Dict[str, Tensor],
                update_kwargs: Dict[str, Tensor],
                ) -> Tuple[Tensor, Tensor]:
        mean, cov = self.update(input, mean, cov, update_kwargs)
        return self.predict(mean, cov, predict_kwargs)

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        assert len(input.shape) > 1
        if len(input.shape) != 2:
            raise NotImplementedError

        num_groups = input.shape[0]
        if mean.shape[0] != num_groups:
            assert mean.shape[0] == 1
            mean = mean.expand(num_groups, -1)
        if cov.shape[0] != num_groups:
            assert cov.shape[0] == 1
            cov = cov.expand(num_groups, -1, -1)

        H = kwargs['H']
        R = kwargs['R']

        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            new_mean = mean.clone()
            new_cov = cov.clone()
            for groups, val_idx in get_nan_groups(isnan):
                if val_idx is None:
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[groups], mean=mean[groups], cov=cov[groups], H=H[groups], R=R[groups]
                    )
                else:
                    # masks:
                    m1d = torch.meshgrid(groups, val_idx)
                    m2d = torch.meshgrid(groups, val_idx, val_idx)
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[m1d[0], m1d[1]],
                        mean=mean[groups],
                        cov=cov[groups],
                        H=H[m1d[0], m1d[1]],
                        R=R[m2d[0], m2d[1], m2d[2]]
                    )
            return new_mean, new_cov
        else:
            return self._update(input=input, mean=mean, cov=cov, H=H, R=R)

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, H: Tensor, R: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError
