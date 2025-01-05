from typing import Tuple, Optional, Sequence, List, TYPE_CHECKING, Union, Dict

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import torch
from torch import Tensor
from torch.distributions import MultivariateNormal, Binomial

from torchcast.covariance import Covariance
from torchcast.covariance.util import mini_cov_mask
from torchcast.internals.utils import identity
from torchcast.kalman_filter import KalmanFilter
from torchcast.kalman_filter.ekf import EKFStep, EKFPredictions
from torchcast.process import Process
from torchcast.utils import conf2bounds, class_or_instancemethod

if TYPE_CHECKING:
    from pandas import DataFrame


class BinomialStep(EKFStep):
    def __init__(self, binary_idx: Optional[Sequence[int]] = None):
        super().__init__()
        self.binary_idx = binary_idx

    def _adjust_h(self, mean: Tensor, H: Tensor, kwargs: Dict[str, Tensor]) -> Tensor:
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
        assert not ((H != 0).sum(-2) > 1).any(), "BinomialFilter does not support the provided measurement-matrix"
        num_obs = kwargs['num_obs']
        if (num_obs != 1).any():
            raise NotImplementedError
        all_idx = list(range(H.shape[-1]))
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        h_dot_state = (H @ mean.unsqueeze(-1)).squeeze(-1)
        adjustment = torch.ones_like(h_dot_state)
        numer = torch.exp(-h_dot_state[..., binary_idx])
        denom = (torch.exp(-h_dot_state[..., binary_idx]) + 1) ** 2
        adjustment[..., binary_idx] = numer / denom
        return H * adjustment.unsqueeze(-1)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor], kwargs: dict[str, Tensor]) -> Tensor:
        all_idx = list(range(R.shape[-1]))
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        newR = torch.zeros_like(R)
        # for binary measures, var==mean:
        newR[..., binary_idx, binary_idx] = measured_mean[..., binary_idx] * (1 - measured_mean[..., binary_idx])

        if gaussian_idx:
            # for gaussian measures, would be much more readable code if we just looped over the gaussian dims
            # and modified newR in-place. but now that newR has grad, don't want to modify in-place.
            # so we use masking trick
            gaussian_idx = torch.as_tensor(gaussian_idx, device=measured_mean.device)
            gaussian_cidx = torch.meshgrid(torch.arange(newR.shape[0]), gaussian_idx, gaussian_idx, indexing='ij')
            expand_mask = mini_cov_mask(rank=len(all_idx), empty_idx=binary_idx)  # todo: cache?
            gaussianR = expand_mask @ R[gaussian_cidx] @ expand_mask.transpose(-1, -2)
            newR = newR + gaussianR

        return newR

    def _adjust_measurement(self, x: Tensor, kwargs: dict[str, Tensor]) -> Tensor:
        all_idx = list(range(x.shape[-1]))
        binary_idx = list(all_idx if self.binary_idx is None else self.binary_idx)
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # again some awkwardness due to avoiding in-place on tensors with grad
        binary_out = torch.zeros_like(x)
        binary_out[..., binary_idx] = torch.sigmoid(x[..., binary_idx])
        gaussian_out = torch.zeros_like(x)
        gaussian_out[..., gaussian_idx] = x[..., gaussian_idx]
        return binary_out + gaussian_out


class BinomialFilter(KalmanFilter):
    ss_step_cls = BinomialStep

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
    def ss_step(self) -> 'BinomialStep':
        return self.ss_step_cls(binary_idx=[idx for idx, m in enumerate(self.measures) if m in self.binary_measures])

    def _get_measure_scaling(self) -> Tensor:
        # TODO: less code duplication?
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        measure_var = mcov.diagonal(dim1=-2, dim2=-1)
        multi = torch.zeros(mcov.shape[0:-2] + (self.state_rank,), dtype=mcov.dtype, device=mcov.device)
        for pid, process in self.processes.items():
            pidx = self.process_to_slice[pid]
            if process.measure in self.binary_measures:
                multi[..., slice(*pidx)] = 1.0
            else:
                multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
        assert (multi > 0).all()
        return multi

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'BinomialPredictions':
        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        preds = BinomialPredictions(
            *preds,
            R=kwargs.pop('R'),
            H=kwargs.pop('H'),
            model=self,
            binary_measures=self.binary_measures,
            **kwargs
        )
        return preds

    @torch.jit.ignore()
    def set_initial_values(self, y: Tensor, ilinks: Optional[dict[str, callable]] = None, verbose: bool = True):
        ilinks = {m: (logit if m in self.binary_measures else identity) for m in self.measures}
        return super().set_initial_values(y=y, ilinks=ilinks, verbose=verbose)

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        num_obs = kwargs.pop('num_obs', 1)
        kwargs_per_process = super()._parse_design_kwargs(input=input, out_timesteps=out_timesteps, **kwargs)
        kwargs_per_process['num_obs'] = num_obs
        return kwargs_per_process

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        num_obs = kwargs_per_process.pop('num_obs')
        predict_kwargs, update_kwargs = super()._build_design_mats(
            kwargs_per_process=kwargs_per_process,
            num_groups=num_groups,
            out_timesteps=out_timesteps
        )
        if isinstance(num_obs, int):
            update_kwargs['num_obs'] = [torch.ones(num_groups) * num_obs for _ in range(out_timesteps)]
        else:
            raise NotImplementedError
        return predict_kwargs, update_kwargs


class BinomialPredictions(EKFPredictions):

    def __init__(self,
                 state_means: Sequence[Tensor],
                 state_covs: Sequence[Tensor],
                 R: Sequence[Tensor],
                 H: Sequence[Tensor],
                 model: Union['StateSpaceModel', 'StateSpaceModelMetadata'],
                 binary_measures: Sequence[str],
                 num_obs: Union[int, Tensor],
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
        self.num_obs = torch.stack(num_obs, -1)
        self.binary_measures = binary_measures

    @property
    def binary_idx(self) -> Sequence[int]:
        return [idx for idx, m in enumerate(self.measures) if m in self.binary_measures]

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        num_samples = 600  # TODO: how to allow user to customize?
        torch.manual_seed(0)  # TODO: this is a hack; instead need to sample white noise once per training session

        group_idx = torch.arange(covs.shape[0], dtype=torch.int)
        binary_idx = torch.as_tensor(self.binary_idx, dtype=torch.int)
        gauss_idx = torch.as_tensor([i for i in range(covs.shape[-1]) if i not in self.binary_idx], dtype=torch.int)
        if len(gauss_idx):
            gauss_cidx = torch.meshgrid(group_idx, gauss_idx, gauss_idx, indexing='ij')
            # super of BinomialPredictions is NotImplemented; we use super of EKFPredictions
            gauss_lp = super(EKFPredictions, self)._log_prob(
                obs=obs[..., gauss_idx], means=means[..., gauss_idx], covs=covs[gauss_cidx]
            )
        else:
            gauss_lp = 0

        if len(binary_idx):
            binary_cidx = torch.meshgrid(group_idx, binary_idx, binary_idx, indexing='ij')
            state_mvnorm = MultivariateNormal(means[..., self.binary_idx], covs[binary_cidx], validate_args=False)
            mc = state_mvnorm.rsample(sample_shape=(num_samples,))
            if len(self.num_obs.shape) == 2:
                nobs = self.num_obs.view(-1).unsqueeze(-1)
            else:
                raise NotImplementedError
            binom = Binomial(total_count=nobs, logits=mc, validate_args=False)
            binary_lp = binom.log_prob(obs[..., self.binary_idx]).mean(0).sum(-1)
        else:
            binary_lp = 0

        return gauss_lp + binary_lp

    @class_or_instancemethod
    def _adjust_measured_mean(cls,
                              x: Union[Tensor, np.ndarray, pd.Series],
                              measure: str,
                              std: Optional[Union[Tensor, np.ndarray, pd.Series]] = None,
                              conf: float = .95) -> Union[Tensor, pd.DataFrame]:
        if isinstance(cls, type):
            raise RuntimeError(f"Cannot call this method as a classmethod for {cls.__name__}, only instance method.")

        x_index = None
        if hasattr(x, 'to_numpy'):
            x_index = x.index if hasattr(x, 'index') else None
            x = x.to_numpy()

        if std is None:
            x = torch.as_tensor(x)
            if measure in cls.binary_measures:
                return torch.sigmoid(x)
            return x

        with torch.no_grad():
            if hasattr(std, 'to_numpy'):
                std = std.to_numpy()

            lower, upper = conf2bounds(x, std, conf=conf)
            # todo: this is only state-uncertainty, should we also include measurement-uncertainty: p*(1-p)?
            out = pd.DataFrame({'mean': x, 'lower': lower, 'upper': upper}, index=None if x_index is None else x_index)
            if measure in cls.binary_measures:
                out.loc[:, ['mean', 'lower', 'upper']] = expit(out)
            return out


def main(num_groups: int = 10, num_timesteps: int = 200, bias: float = -1):
    from torchcast.process import LocalLevel, LocalTrend
    from torchcast.utils import TimeSeriesDataset
    import pandas as pd
    from plotnine import geom_line, aes, ggtitle
    torch.manual_seed(1234)

    measures = ['dim1', 'dim2']
    latent = (
            torch.randn((num_groups, 1, len(measures)))
            + torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, len(measures))), dim=1)
            + bias
    )
    y = []
    for i, m in enumerate(measures):
        if i:
            y.append(torch.distributions.Normal(loc=latent[..., i], scale=.5).sample())
        else:
            y.append(torch.distributions.Binomial(logits=latent[..., 0]).sample())
    y = torch.stack(y, dim=-1)
    # first tensor in dataset is observed
    # second tensor is ground truth
    dataset = TimeSeriesDataset(
        y,
        latent,
        group_names=[f'group_{i}' for i in range(num_groups)],
        start_times=[pd.Timestamp('2023-01-01')] * num_groups,
        measures=[measures, [x.replace('dim', 'latent') for x in measures]],
        dt_unit='D'
    )

    bf = BinomialFilter(
        processes=[LocalTrend(id=f'trend_{m}', measure=m) for m in measures],
        measures=measures,
        binary_measures=['dim1']
    )

    bf.fit(dataset.tensors[0])
    preds = bf(dataset.tensors[0])
    df_preds = preds.to_dataframe(dataset)
    df_latent = (dataset.to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='latent')
                 .assign(measure=lambda _df: _df['measure'].str.replace('latent', 'dim')))
    _is_binary = df_latent['measure'] == measures[0]
    df_latent.loc[_is_binary, 'latent'] = expit(df_latent.loc[_is_binary, 'latent'])

    df_plot = df_preds.merge(df_latent, how='left', on=['group', 'time', 'measure'])
    for g, _df in df_plot.query("group.isin(group.drop_duplicates().sample(5))").groupby('group'):
        print(
            preds.plot(_df)
            + geom_line(aes(y='latent'), color='purple')
            + ggtitle(g)
        )


if __name__ == '__main__':
    main()
