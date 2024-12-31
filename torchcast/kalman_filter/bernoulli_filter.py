import functools
from typing import Tuple, Optional, Sequence, List, TYPE_CHECKING, Union

import numpy as np
import pandas as pd
from scipy.special import expit

import torch
from torch import Tensor, nn

from torchcast.covariance import Covariance
from torchcast.covariance.util import mini_cov_mask
from torchcast.kalman_filter import KalmanFilter
from torchcast.kalman_filter.ekf import EKFStep, EKFPredictions
from torchcast.process import Process
from torchcast.state_space.ss_step import StateSpaceStep
from torchcast.utils import conf2bounds

if TYPE_CHECKING:
    from pandas import DataFrame

sigmoid = nn.Sigmoid()


class BernoulliStep(EKFStep):
    def __init__(self, binary_idx: Optional[Sequence[int]] = None):
        super().__init__()
        self.binary_idx = binary_idx

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
        all_idx = list(range(H.shape[-1]))  # TODO: confirm
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        h_dot_state = (H @ mean.unsqueeze(-1)).squeeze(-1)
        adjustment = torch.ones_like(h_dot_state)
        numer = torch.exp(-h_dot_state[..., binary_idx])
        denom = (torch.exp(-h_dot_state[..., binary_idx]) + 1) ** 2
        adjustment[..., binary_idx] = numer / denom
        return H * adjustment.unsqueeze(-1)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor]) -> Tensor:
        all_idx = list(range(R.shape[-1]))
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        newR = torch.zeros_like(R)
        # for binary measures, var==mean:
        newR[..., binary_idx, binary_idx] = measured_mean[..., binary_idx] * (1 - measured_mean[..., binary_idx])

        if gaussian_idx:
            # for gaussian measures, would be much more readable code if we just looped over the gaussian dims
            # and modified newR in-place. but now that newR has grad, that's no good. so we use masking trick
            gaussian_idx = torch.as_tensor(gaussian_idx, device=measured_mean.device)
            gaussian_mask = torch.meshgrid(torch.arange(newR.shape[0]), gaussian_idx, gaussian_idx, indexing='ij')
            expand_mask = mini_cov_mask(rank=len(all_idx), empty_idx=binary_idx)  # todo: cache?
            gaussianR = expand_mask @ R[gaussian_mask] @ expand_mask.transpose(-1, -2)
            newR = newR + gaussianR

        return newR

    def _adjust_measurement(self, x: Tensor) -> Tensor:
        all_idx = list(range(x.shape[-1]))
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # again some awkwardness due to avoiding in-place on tensors with grad
        binary_out = torch.zeros_like(x)
        binary_out[..., binary_idx] = sigmoid(x[..., binary_idx])
        gaussian_out = torch.zeros_like(x)
        gaussian_out[..., gaussian_idx] = x[..., gaussian_idx]
        return binary_out + gaussian_out


class BernoulliFilter(KalmanFilter):
    ss_step_cls = BernoulliStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 binary_measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None):

        if isinstance(measures, str):
            raise ValueError(f"`measures` should be a list of strings not a string.")
        self.binary_measures = measures if binary_measures is None else binary_measures
        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=Covariance(
                rank=len(measures),
                empty_idx=[i for i, m in enumerate(measures) if m in self.binary_measures]
            ),
        )

    @property
    def ss_step(self) -> StateSpaceStep:
        return self.ss_step_cls(binary_idx=[idx for idx, m in enumerate(self.measures) if m in self.binary_measures])

    def _get_measure_scaling(self) -> Tensor:
        if set(self.binary_measures) != set(self.measures):
            raise NotImplementedError("TODO")
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

        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        preds = BernoulliPredictions(
            *preds,
            R=kwargs.pop('R'),
            H=kwargs.pop('H'),
            model=self,
            binary_idx=[idx for idx, m in enumerate(self.measures) if m in self.binary_measures],
            **kwargs
        )
        return preds


_warn_once = {}


class BernoulliPredictions(EKFPredictions):

    def __init__(self,
                 state_means: Sequence[Tensor],
                 state_covs: Sequence[Tensor],
                 R: Sequence[Tensor],
                 H: Sequence[Tensor],
                 model: Union['StateSpaceModel', 'StateSpaceModelMetadata'],
                 binary_idx: Sequence[int],
                 update_means: Optional[Sequence[Tensor]] = None,
                 update_covs: Optional[Sequence[Tensor]] = None):
        super().__init__(
            state_means=state_means,
            state_covs=state_covs,
            R=R,
            H=H,
            model=model,
            update_means=update_means,
            update_covs=update_covs
        )
        self.binary_idx = binary_idx

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        num_samples = 600  # TODO: how to allow user to customize?
        torch.manual_seed(0)  # TODO: this is a hack; instead need to sample white noise once per training session
        if len(self.binary_idx) < obs.shape[-1]:
            raise NotImplementedError("TODO")
        samples = torch.distributions.MultivariateNormal(means, covs).rsample(sample_shape=(num_samples,))
        return torch.distributions.Bernoulli(logits=samples).log_prob(obs).mean(0).sum(-1)

    @classmethod
    def _adjust_measured_mean(cls,
                              x: Union[Tensor, np.ndarray, pd.Series],
                              std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                              conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        x_index = None
        if hasattr(x, 'to_numpy'):
            x_index = x.index if hasattr(x, 'index') else None
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
            }, index=None if x_index is None else x_index)


def main(num_groups: int = 10, num_timesteps: int = 200, bias: int = 0):
    from torchcast.process import LocalLevel, LocalTrend
    from torchcast.utils import TimeSeriesDataset
    import pandas as pd
    from plotnine import geom_line, aes, facet_wrap
    torch.manual_seed(1234)

    measures = ['dim1', 'dim2']
    logits = (
        torch.randn((num_groups, 1, len(measures)))
        + torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, len(measures))), dim=1)
        + bias
    )
    y = torch.distributions.Bernoulli(logits=logits).sample()
    dataset = TimeSeriesDataset(
        y,
        sigmoid(logits),
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
