from typing import Tuple, Dict, Optional

import torch
from torch import Tensor

from torchcast.internals.utils import get_nan_groups


class StateSpaceStep(torch.nn.Module):
    """
    Base-class for modules that handle predict/update within a state-space model.
    """

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

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
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
                m, c = self._update(
                    input=masked_input,
                    mean=mean[groups],
                    cov=cov[groups],
                    kwargs=masked_kwargs
                )
                new_mean[groups] = m
                if c is None:
                    c = 0
                new_cov[groups] = c
            return new_mean, new_cov
        else:
            return self._update(input=input, mean=mean, cov=cov, kwargs=kwargs)

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor],
                   kwargs_dims: Optional[Dict[str, int]]) -> Tuple[Tensor, Dict[str, Tensor]]:
        new_kwargs = kwargs.copy()
        if val_idx is None:
            for k in kwargs_dims:
                new_kwargs[k] = kwargs[k][groups]
            return input[groups], new_kwargs
        else:
            m1d = torch.meshgrid(groups, val_idx, indexing='ij')  # todo: why not just [groups][val_idx]?
            m2d = torch.meshgrid(groups, val_idx, val_idx, indexing='ij')
            masked_input = input[m1d[0], m1d[1]]
            for k, dim in kwargs_dims.items():
                if dim == 0:
                    new_kwargs[k] = kwargs[k][groups]
                elif dim == 1:
                    new_kwargs[k] = kwargs[k][m1d[0], m1d[1]]
                elif dim == 2:
                    new_kwargs[k] = kwargs[k][m2d[0], m2d[1], m2d[2]]
                else:
                    raise ValueError(f"Invalid dim ({dim}) for {k}")
            return masked_input, new_kwargs
