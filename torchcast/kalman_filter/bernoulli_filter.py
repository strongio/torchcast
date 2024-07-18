import functools
from typing import Dict, Tuple, Optional, Sequence, Iterable, List, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch import Tensor, nn

from functools import cached_property

from torchcast.covariance import Covariance
from torchcast.kalman_filter import KalmanFilter
from torchcast.kalman_filter.ekf import EKFStep, EKFPredictions
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel
from torchcast.utils import conf2bounds

if TYPE_CHECKING:
    from pandas import DataFrame

sigmoid = nn.Sigmoid()


class BernoulliStep(EKFStep):
    def _adjust_h(self, mean: Tensor, H: Tensor) -> Tensor:
        """
        >>> import sympy
        >>> from sympy import exp, Matrix
        >>>
        >>>  def full_like(x, value):
        >>>     nrow, ncol = x.shape
        >>>     return sympy.Matrix([[value]*ncol]*nrow)
        >>>
        >>> def ones_like(x):
        >>>     return full_like(x, 1)
        >>>
        >>> def sigmoid(x):
        >>>     return (ones_like(x) + exp(-x)) ** -1
        >>>
        >>> sympy.init_printing(use_latex='mathjax')
        >>>
        >>> x1, x2, x3 = sympy.symbols("x, x', y")
        >>>
        >>> H = sympy.Matrix([[1, 0, 1]])
        >>>
        >>> state = sympy.Matrix([x1, x2, x3])
        >>> mmean_orig = H @ state
        >>> measured_mean = sigmoid(mmean_orig)
        >>>
        >>> J = measured_mean.jacobian(state)
        >>>
        >>> def jacob(h_dot_state):
        >>>     numer = exp(-h_dot_state)
        >>>     denom = (exp(-h_dot_state) + ones_like(h_dot_state)) ** 2
        >>>     return numer * denom ** -1  # sympy being weird, numer/denom doesn't work
        >>>
        >>> sympy.matrices.dense.matrix_multiply_elementwise(
        >>>     full_like(H, jacob(mmean_orig)),
        >>>     H
        >>> )
        """
        # this assert should not fail, b/c all processes only support a single measure
        assert not ((H != 0).sum(-2) > 1).any(), "BernoulliFilter does not support the provided measurement-matrix"
        h_dot_state = H @ mean.unsqueeze(-1)
        numer = torch.exp(-h_dot_state)
        denom = (torch.exp(-h_dot_state) + 1) ** 2
        return H * (numer / denom)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor]) -> Tensor:
        var = measured_mean * (1 - measured_mean)
        return torch.diag_embed(var)  # variance = mean

    def _adjust_measurement(self, x: Tensor) -> Tensor:
        return sigmoid(x)


class BernoulliFilter(KalmanFilter):
    ss_step_cls = BernoulliStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None):

        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=Covariance(
                rank=len(measures),
                empty_idx=[i for i, m in enumerate(measures)]  # all zero (future: support gaussian, some bern)
            ),
        )

    def _get_measure_scaling(self) -> Tensor:
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        return torch.ones((self.state_rank,), dtype=mcov.dtype, device=mcov.device)

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'BernoulliPredictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """
        kwargs.update({
            'state_means': preds[0],
            'state_covs': preds[1],
            'model': self
        })
        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        return BernoulliPredictions(**kwargs)

    @torch.jit.ignore()
    def set_initial_values(self, y: Tensor, n: int, ilink: Optional[callable] = None, verbose: bool = True):
        if n is True:
            # use a different default, only one timestep is too stringent
            num_timesteps = y.shape[1]
            n = max(int(num_timesteps * 0.10), 1)
        return super().set_initial_values(y=y, n=n, ilink=functools.partial(torch.logit, eps=.001), verbose=verbose)


_warn_once = {}


class BernoulliPredictions(EKFPredictions):

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        num_samples = 600  # TODO: how to allow user to customize?
        torch.manual_seed(0)  # TODO: this is a hack; instead need to sample white noise once per training session
        samples = super()._sample(
            means=means,
            covs=covs,
            sample_shape=(num_samples,)
        )
        return torch.distributions.Bernoulli(probs=samples).log_prob(obs).mean(0).sum(-1)

    def _sample(self, means: Tensor, covs: Tensor, sample_shape=torch.Size()) -> Tensor:
        samples = super()._sample(
            means=means,
            covs=covs,
            sample_shape=sample_shape
        )
        return torch.distributions.Bernoulli(probs=samples).sample()

    @classmethod
    def inverse_transform(cls,
                          x: Union[Tensor, np.ndarray, pd.Series],
                          std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                          conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        if hasattr(x, 'to_numpy'):
            x = x.to_numpy()

        if std is None:
            return sigmoid(torch.as_tensor(x))

        with torch.no_grad():
            if hasattr(std, 'to_numpy'):
                std = std.to_numpy()

            lower, upper = conf2bounds(x, std, conf=conf)
            # todo: this is only state-uncertainty, should we also include measurement-uncertainty: p*(1-p)?
            return pd.DataFrame({
                'mean': expit(x),
                'lower': expit(lower),
                'upper': expit(upper)
            }, index=x.index if hasattr(x, 'index') else None)


def main(num_groups: int = 6, num_timesteps: int = 200, bias: int = 0):
    from torchcast.process import LocalLevel, LocalTrend
    from torchcast.utils import TimeSeriesDataset
    import pandas as pd
    from plotnine import geom_line, aes, facet_wrap

    measures = ['dim1']
    probs = sigmoid(
        torch.randn((num_groups, 1, len(measures)))
        + torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, len(measures))), dim=1)
        + bias
    )
    y = torch.distributions.Bernoulli(probs=probs).sample()
    dataset = TimeSeriesDataset(
        y,
        probs,
        group_names=[f'group_{i}' for i in range(num_groups)],
        start_times=[pd.Timestamp('2023-01-01')] * num_groups,
        measures=[measures, [x.replace('dim', 'probs') for x in measures]],
        dt_unit='D'
    )

    bf = BernoulliFilter(
        processes=[LocalLevel(id=f'trend_{m}', measure=m) for m in measures],
        measures=measures
    )
    bf.fit(dataset.tensors[0])
    preds = bf(dataset.tensors[0])
    df = (preds
          .to_dataframe(dataset)
          .merge(dataset
                 .to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='probs')
                 .assign(measure=lambda df: df['measure'].str.replace('probs', 'dim')),
                 how='left',
                 on=['group', 'time', 'measure']))
    p = (
            preds.plot(df, max_num_groups=4)
            + geom_line(aes(y='probs'), color='purple')
    )
    if len(measures) == 1:
        p += facet_wrap('group')
    print(p)


if __name__ == '__main__':
    main()
