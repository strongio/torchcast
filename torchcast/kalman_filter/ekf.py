from typing import Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from .kalman_filter import KalmanStep
from ..state_space import Predictions
from ..utils import TimeSeriesDataset


class EKFStep(KalmanStep):
    """
    Implements update for the extended kalman-filter. Currently limited:

    1. Does not implement any special ``predict``, so no custom functions for transition (only measurement).
    2. Assumes the custom measure function takes the form ``custom_fun(H @ state)``.

    This means that currently, this is primarily useful for approximating non-gaussian measures (e.g. poisson,
     binomial) via a link function.
    """

    def _adjust_h(self, mean: Tensor, H: Tensor) -> Tensor:
        return H

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor]) -> Tensor:
        assert R is not None
        return R

    def _adjust_measurement(self, x: Tensor) -> Tensor:
        return x

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        if (kwargs['outlier_threshold'] != 0).any():
            raise NotImplementedError("Outlier rejection is not yet supported for EKF")

        orig_H = kwargs['H']
        h_dot_state = (orig_H @ mean.unsqueeze(-1)).squeeze(-1)
        kwargs['measured_mean'] = self._adjust_measurement(h_dot_state)
        kwargs['H'] = self._adjust_h(mean, orig_H)
        kwargs['R'] = self._adjust_r(kwargs['measured_mean'], kwargs.get('R', None))

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs=kwargs
        )


class EKFPredictions(Predictions):
    @classmethod
    def inverse_transform(cls,
                          x: Union[Tensor, np.ndarray, pd.Series],
                          std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                          conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        raise NotImplementedError

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        raise NotImplementedError

    def _sample(self, means: Tensor, covs: Tensor, sample_shape=torch.Size()) -> Tensor:
        samples = super()._sample(
            means=means,
            covs=covs,
            sample_shape=sample_shape
        )
        return self.inverse_transform(samples)

    def _inverse_transform_tensors(self, means, covs) -> np.ndarray:
        means = means.numpy()
        stds = torch.diagonal(covs, dim1=-1, dim2=-2).sqrt().numpy()
        out = []
        for i, m in enumerate(self.measures):
            out.append(self.link(means[..., i], stds[..., i]))
        return np.stack(out, -1)

    def __array__(self) -> np.ndarray:
        with torch.no_grad():
            return self._inverse_transform_tensors(self.means, self.covs)

    def to_dataframe(self,
                     dataset: Union[TimeSeriesDataset, dict],
                     type: str = 'predictions',
                     group_colname: str = 'group',
                     time_colname: str = 'time',
                     conf: Optional[float] = .95,
                     multi: Optional[float] = None) -> pd.DataFrame:
        df = super().to_dataframe(
            dataset=dataset,
            type=type,
            group_colname=group_colname,
            time_colname=time_colname,
            conf=None,
            multi=multi
        )
        df[['mean', 'lower', 'upper']] = self.inverse_transform(df['mean'], df.pop('std'), conf=conf)
        return df

    @classmethod
    def plot(cls,
             df: pd.DataFrame,
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> pd.DataFrame:

        if 'upper' not in df.columns and 'std' in df.columns:
            df[['mean', 'lower', 'upper']] = cls.inverse_transform(df['mean'], df['std'])

        return super().plot(
            df=df,
            group_colname=group_colname,
            time_colname=time_colname,
            max_num_groups=max_num_groups,
            split_dt=split_dt,
            **kwargs
        )
