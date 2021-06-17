from typing import Type, Tuple, Dict, Optional

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

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Handles validation and masking of missing values. Core update logic implemented in ``_update()``.
        """
        assert len(input.shape) > 1
        assert len(input.shape) == 2

        num_groups = input.shape[0]
        if mean.shape[0] != num_groups:
            assert mean.shape[0] == 1
            mean = mean.expand(num_groups, -1)
        if cov.shape[0] != num_groups:
            assert cov.shape[0] == 1
            cov = cov.expand(num_groups, -1, -1)

        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            new_mean = mean.clone()
            new_cov = cov.clone()
            for groups, val_idx in get_nan_groups(isnan):
                masked_input, masked_kwargs = self._mask_mats(groups, val_idx, input=input, kwargs=kwargs)
                new_mean[groups], new_cov[groups] = self._update(
                    input=masked_input, mean=mean[groups], cov=cov[groups], kwargs=masked_kwargs
                )
            return new_mean, new_cov
        else:
            return self._update(input=input, mean=mean, cov=cov, kwargs=kwargs)

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        raise NotImplementedError
