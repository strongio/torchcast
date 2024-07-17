import functools
from typing import Dict, Tuple, Optional, Type, Sequence, Iterable, List, TYPE_CHECKING
from warnings import warn

import numpy as np
import torch
from scipy.special import expit, logit
from torch import Tensor, nn

from backports.cached_property import cached_property
from torch.nn import Sigmoid

from torchcast.covariance import Covariance
from torchcast.kalman_filter.ekf import EKFStep
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel, Predictions

if TYPE_CHECKING:
    from pandas import DataFrame

sigmoid = Sigmoid()


class BernoulliStep(EKFStep):
    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.Bernoulli

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
        >>>     return numer * denom ** -1
        >>>
        >>> sympy.matrices.dense.matrix_multiply_elementwise(
        >>>     full_like(H, jacob(mmean_orig)),
        >>>     H
        >>> )
        """
        h_dot_state = H @ mean.unsqueeze(-1)  # 1x3 @ 2x1 = 3x1
        numer = torch.exp(-h_dot_state)
        denom = (torch.exp(-h_dot_state) + 1) ** 2
        return H * (numer / denom)

    def _adjust_r(self, measured_mean: Tensor, R: Optional[Tensor]) -> Tensor:
        var = measured_mean * (1 - measured_mean)
        return torch.diag_embed(var)  # variance = mean

    def _link(self, measured_mean: Tensor) -> Tensor:
        return sigmoid(measured_mean)


class BernoulliFilter(StateSpaceModel):
    ss_step_cls = BernoulliStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None):

        initial_covariance = Covariance.from_processes(processes, cov_type='initial')

        if process_covariance is None:
            process_covariance = Covariance.from_processes(processes, cov_type='process')

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=None,
        )
        self.process_covariance = process_covariance.set_id('process_covariance')
        self.initial_covariance = initial_covariance.set_id('initial_covariance')

    @torch.jit.ignore()
    def design_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'process_covariance', self.process_covariance
        yield 'initial_covariance', self.initial_covariance

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        Fs, Hs = self._build_transition_and_measure_mats(kwargs_per_process, num_groups, out_timesteps)

        # process-variance:
        Qs = self.process_covariance(
            kwargs_per_process.get('process_covariance', {}), num_groups=num_groups, num_times=out_timesteps
        )
        Qs = Qs.unbind(1)

        predict_kwargs = {'F': Fs, 'Q': Qs}
        update_kwargs = {'H': Hs}
        return predict_kwargs, update_kwargs

    def _get_measure_scaling(self) -> Tensor:
        pcov = self.process_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        multi = torch.ones((self.state_rank,), dtype=pcov.dtype, device=pcov.device)
        return multi

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[List[Tensor], List[Tensor]],
                              updates: Optional[Tuple[List[Tensor], List[Tensor]]] = None,
                              **kwargs) -> 'Predictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """

        kwargs = {
            'state_means': preds[0],
            'state_covs': preds[1],
            'H': kwargs['H'],
            'R': None,
            'model': self
        }
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


class BernoulliPredictions(Predictions):

    @classmethod
    def observe(cls,
                state_means: Tensor,
                state_covs: Tensor,
                R: Optional[Tensor],
                H: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: monte-carlo?
        means, covs = super().observe(
            state_means=state_means,
            state_covs=state_covs,
            R=R,
            H=H
        )
        means = sigmoid(means)

        return means, covs

    @classmethod
    def plot(cls,
             df: 'DataFrame',
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> 'DataFrame':
        if not _warn_once.get('bern_plot', False):
            _warn_once['bern_plot'] = True
            warn(
                "Bernoulli implementation is experimental. Uncertainty intervals in plots are not guaranteed."
            )
        return super().plot(
            df=df,
            group_colname=group_colname,
            time_colname=time_colname,
            max_num_groups=max_num_groups,
            split_dt=split_dt,
            **kwargs
        )

    @cached_property
    def R(self) -> torch.Tensor:
        means, _ = self.observe(self.state_means, self.state_covs, R=0.0, H=self.H)
        var = means * (1 - means)
        return torch.diag_embed(var)

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        if not _warn_once.get('bern_log_prob', False):
            _warn_once['bern_log_prob'] = True
            warn(
                "Bernoulli implementation is experimental. Currently log-prob will ignore state-covariance."
            )

        # TODO: use monte-carlo instead.
        out = self.distribution_cls(means.squeeze(-1), validate_args=False).log_prob(obs.squeeze(-1))
        return out

    def sample(self) -> Tensor:
        raise NotImplementedError

    @classmethod
    def _get_quantiles(cls, mean, std, conf: float, observed: bool) -> tuple:
        if not observed:
            return super()._get_quantiles(mean=mean, std=std, conf=conf, observed=observed)
        lower_logit, upper_logit = super()._get_quantiles(
            mean=logit(mean),
            std=std,
            conf=conf,
            observed=observed
        )
        return expit(lower_logit), expit(upper_logit)


def main():
    from torchcast.process import LocalTrend
    import pandas as pd

    from torchcast.utils import TimeSeriesDataset
    probs = sigmoid(torch.randn((5, 1, 1)) + torch.cumsum(.1 * torch.randn((5, 100, 1)), dim=1) - 3)
    y = torch.distributions.Bernoulli(probs=probs).sample()
    dataset = TimeSeriesDataset(
        y,
        probs,
        group_names=['a', 'b', 'c', 'd', 'e'],
        start_times=[pd.Timestamp('2023-01-01')] * 5,
        measures=[['nonzero'], ['probs']],
        dt_unit='D'
    )

    bf = BernoulliFilter(
        processes=[LocalTrend(id='trend')],
        measures=['nonzero']
    )
    bf.fit(dataset.tensors[0])
    preds = bf(dataset.tensors[0])
    df = preds.to_dataframe(dataset).merge(dataset.to_dataframe().drop(columns=['nonzero']), how='left')
    from plotnine import geom_line, aes
    print(preds.plot(df, max_num_groups=4) + geom_line(aes(y='probs'), color='purple'))


if __name__ == '__main__':
    main()
