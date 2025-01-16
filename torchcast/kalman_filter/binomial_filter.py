from typing import Tuple, Optional, Sequence, List, TYPE_CHECKING, Union, Dict

import numpy as np
import pandas as pd
from scipy.special import expit, logit

import torch
from torch import Tensor
from torch.distributions import Binomial

from torchcast.covariance import Covariance
from torchcast.covariance.util import mini_cov_mask
from torchcast.internals.utils import identity, class_or_instancemethod
from torchcast.kalman_filter import KalmanFilter
from torchcast.kalman_filter.ekf import EKFStep, EKFPredictions
from torchcast.process import Process
from torchcast.state_space.predictions import conf2bounds

if TYPE_CHECKING:
    from pandas import DataFrame


class BinomialStep(EKFStep):
    def __init__(self, binary_idx: Optional[Sequence[int]] = None):
        super().__init__()
        self.binary_idx = binary_idx

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:

        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = list(range(input.shape[-1]))
        if (input[:, binary_idx] < 0).any():
            raise ValueError("BinomialFilter does not support negative inputs.")
        if (input[:, binary_idx] > 1).any():
            raise ValueError("BinomialFilter expects inputs that are proportions (0 <= p <= 1).")

        return super()._update(
            input=input,
            mean=mean,
            cov=cov,
            kwargs=kwargs
        )

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor],
                   kwargs_dims: Optional[Dict[str, int]] = None) -> Tuple[Tensor, Dict[str, Tensor]]:
        masked_input, new_kwargs = super()._mask_mats(
            groups=groups,
            val_idx=val_idx,
            input=input,
            kwargs=kwargs
        )

        if self.binary_idx is not None and val_idx is not None:
            new_kwargs['binary_idx'] = torch.as_tensor(
                [new_idx for new_idx, og_idx in enumerate(val_idx) if og_idx in self.binary_idx],
                dtype=torch.int,
                device=input.device
            )
        if 'num_obs' in kwargs:
            new_kwargs['num_obs'] = kwargs['num_obs'][groups]
            if val_idx is not None:
                _keep_idx = [i for i, bidx in enumerate(self.binary_idx) if bidx in val_idx]
                new_kwargs['num_obs'] = new_kwargs['num_obs'][:, _keep_idx]

        return masked_input, new_kwargs

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
        all_idx = list(range(H.shape[-1]))
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        h_dot_state = (H @ mean.unsqueeze(-1)).squeeze(-1)
        adjustment = torch.ones_like(h_dot_state)
        binary_h_dot_state = h_dot_state[..., binary_idx].clamp(-10, 10)
        numer = torch.exp(-binary_h_dot_state)
        denom = (torch.exp(-binary_h_dot_state) + 1) ** 2
        adjustment[..., binary_idx] = numer / denom
        return H * adjustment.unsqueeze(-1)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor], kwargs: dict[str, Tensor]) -> Tensor:
        all_idx = list(range(R.shape[-1]))
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # binomial variance:
        newR = torch.zeros_like(R)
        binary_measured_mean = measured_mean[..., binary_idx]
        newR[..., binary_idx, binary_idx] = (
                kwargs['num_obs'] * binary_measured_mean * (1 - binary_measured_mean)
        )

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
        binary_idx = kwargs.get('binary_idx', self.binary_idx)
        if binary_idx is None:
            binary_idx = all_idx
        gaussian_idx = [idx for idx in all_idx if idx not in binary_idx]

        # again some awkwardness due to avoiding in-place on tensors with grad
        binary_out = torch.zeros_like(x)
        binary_out[..., binary_idx] = torch.sigmoid(x[..., binary_idx].clamp(-5, 5))
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
        if isinstance(binary_measures, str):
            raise ValueError(f"`binary_measures` should be a list of strings not a string.")
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
            # todo: device
            update_kwargs['num_obs'] = [torch.ones(num_groups, len(self.binary_measures)) * num_obs] * out_timesteps
        else:
            raise NotImplementedError
        return predict_kwargs, update_kwargs

    def fit(self,
            *args,
            mc_samples: int,
            **kwargs):
        """
        :param mc_samples: Number of samples to draw for MC approximation to binomial likelihood.
        :param kwargs: Additional keyword arguments, see `func:torchcast.kalman_filter.KalmanFilter.fit`.
        """
        device = next(iter(self.parameters())).device
        kwargs['prediction_kwargs'] = kwargs.get('prediction_kwargs', {})
        kwargs['prediction_kwargs']['white_noise'] = torch.randn(
            (mc_samples, len(self.binary_measures)),
            device=device
        )
        return super().fit(*args, **kwargs)


class BinomialPredictions(EKFPredictions):

    def __init__(self,
                 state_means: Sequence[Tensor],
                 state_covs: Sequence[Tensor],
                 R: Sequence[Tensor],
                 H: Sequence[Tensor],
                 model: Union['StateSpaceModel', 'StateSpaceModelMetadata'],
                 binary_measures: Sequence[str],
                 num_obs: List[Tensor],
                 white_noise: Optional[Tensor] = None,
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
        self.num_obs = torch.stack(num_obs, 1)
        self.binary_measures = binary_measures
        self._white_noise = white_noise

    @property
    def white_noise(self) -> Tensor:
        if self._white_noise is None:
            # todo: explain more
            raise RuntimeError("Cannot access `white_noise` attribute because it was not passed at init.")
        return self._white_noise

    def _get_log_prob_kwargs(self, groups: Tensor, valid_idx: Tensor) -> dict:
        white_noise = self.white_noise
        num_obs = self.num_obs.view(-1, len(self.binary_measures))[groups]
        if valid_idx is not None:
            valid_measures = [m for i, m in enumerate(self.measures) if i in valid_idx]
            valid_binary_idx = [i for i, m in enumerate(self.binary_measures) if m in valid_measures]
            num_obs = num_obs[:, valid_binary_idx]
            white_noise = white_noise[:, valid_binary_idx]
        return {
            'measures': [m for i, m in enumerate(self.measures) if valid_idx is None or i in valid_idx],
            'num_obs': num_obs,
            'white_noise': white_noise
        }

    def _log_prob(self,
                  obs: Tensor,
                  means: Tensor,
                  covs: Tensor,
                  num_obs: Tensor,
                  measures: Sequence[str],
                  white_noise: Tensor) -> Tensor:
        group_idx = torch.arange(covs.shape[0], dtype=torch.int)
        binary_idx = torch.as_tensor([i for i, m in enumerate(measures) if m in self.binary_measures], dtype=torch.int)
        gauss_idx = torch.as_tensor([i for i in range(covs.shape[-1]) if i not in binary_idx], dtype=torch.int)
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
            chol = torch.linalg.cholesky(covs[binary_cidx])
            corr_white_noise = chol.unsqueeze(0) @ white_noise.view(-1, 1, len(binary_idx), 1)
            mc_samples = corr_white_noise.squeeze(-1) + means[..., binary_idx].unsqueeze(0)
            binom = Binomial(total_count=num_obs.unsqueeze(0), logits=mc_samples, validate_args=False)
            log_probs = binom.log_prob(obs[..., binary_idx])
            binary_lp = log_probs.mean(0).sum(-1)
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

    measures = ['dim1', 'dim2', 'dim3']
    binary_measures = ['dim1', 'dim3']
    latent = (
            torch.randn((num_groups, 1, len(measures)))
            + torch.cumsum(.05 * torch.randn((num_groups, num_timesteps, len(measures))), dim=1)
            + bias
    )
    # num_groups=10, num_timesteps=200
    # KF no missings: 10/s
    # KF with missings: 3.5/s
    # BF no missings: 4.5/s
    # BF with missings: 2.5/s

    y = []
    for i, m in enumerate(measures):
        if m in binary_measures:
            y.append(torch.distributions.Binomial(logits=latent[..., i]).sample())
        else:
            y.append(torch.distributions.Normal(loc=latent[..., i], scale=.5).sample())
        y[-1][torch.randn((num_groups, num_timesteps)) > .95] = float('nan')  # some random missings
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
        binary_measures=binary_measures
    )

    bf.fit(dataset.tensors[0], mc_samples=64)
    preds = bf(dataset.tensors[0])
    df_preds = preds.to_dataframe(dataset)
    df_latent = (dataset.to_dataframe()
                 .drop(columns=measures)
                 .melt(id_vars=['group', 'time'], var_name='measure', value_name='latent')
                 .assign(measure=lambda _df: _df['measure'].str.replace('latent', 'dim')))
    _is_binary = df_latent['measure'].isin(binary_measures)
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
