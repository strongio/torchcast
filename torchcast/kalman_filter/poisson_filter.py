"""
The :class:`.PoissonFilter` is a :class:`torch.nn.Module` which generates forecasts using the extended kalman-filter.
It uses a softplus link function to link the state-mean to the observed mean, and fixes the measure-variance to equal
the mean.

This class is experimental, and the log-likelihood and predictions do not currently incorporate the state-covariance.

----------
"""
from typing import Dict, Tuple, Optional, Type, Sequence, Iterable, List
from warnings import warn

import torch
from scipy import special as scipy_special
from torch import Tensor, nn

from backports.cached_property import cached_property
from torch.nn import Softplus

from .kalman_filter import KalmanStep
from ..covariance import Covariance
from ..process import Process
from ..state_space import StateSpaceModel, Predictions

POISSON_SMALL_THRESH = 10

softplus = Softplus()


def inverse_softplus(x: torch.Tensor, eps: float = .001) -> torch.Tensor:
    not_too_big = x < 20
    out = x.clone()
    x = x.clamp(min=eps)
    out[not_too_big] = torch.log(torch.exp(x[not_too_big]) - 1)
    return out


class PoissonStep(KalmanStep):
    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.Poisson

    def _update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        orig_H = kwargs['H']
        #
        orig_mmean = (orig_H @ mean.unsqueeze(-1)).squeeze(-1)
        measured_mean = softplus(orig_mmean)
        # variance = mean
        R = torch.diag_embed(measured_mean)

        # use EKF:
        correction = torch.zeros_like(orig_H)
        _do_cor = orig_mmean < POISSON_SMALL_THRESH
        # derivative of softplus:
        correction[_do_cor] = orig_H[_do_cor] / (torch.exp(orig_mmean[_do_cor]) + 1).unsqueeze(-1)
        newH = orig_H - correction

        # standard:
        K = self._kalman_gain(cov=cov, H=newH, R=R)
        resid = input - measured_mean
        new_mean = mean + (K @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = self._covariance_update(cov=cov, K=K, H=newH, R=R)
        return new_mean, new_cov


class PoissonFilter(StateSpaceModel):
    ss_step_cls = PoissonStep

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
        return PoissonPredictions(**kwargs)

    @torch.jit.ignore()
    def set_initial_values(self, y: Tensor, n: int, ilink: Optional[callable] = None, verbose: bool = True):
        if n is True:
            # use a different default, only one timestep is too stringent
            num_timesteps = y.shape[1]
            n = max(int(num_timesteps * 0.10), 1)
        return super().set_initial_values(y=y, n=n, ilink=inverse_softplus, verbose=verbose)


_warn_once = {}


class PoissonPredictions(Predictions):

    @classmethod
    def observe(cls,
                state_means: Tensor,
                state_covs: Tensor,
                R: Optional[Tensor],
                H: Tensor) -> Tuple[Tensor, Tensor]:
        means, covs = super().observe(
            state_means=state_means,
            state_covs=state_covs,
            R=R,
            H=H
        )
        means = softplus(means)
        if _warn_once.get('poisson_predictions', False):
            _warn_once['poisson_predictions'] = True
            warn(
                "Poisson implementation is experimental. Currently, this means that, (1) in plotting, will "
                "over-estimate uncertainty for small values, (2) in log-prob, will ignore state-covariance for small "
                "values."
            )
        return means, covs

    @cached_property
    def R(self) -> torch.Tensor:
        means, _ = self.observe(self.state_means, self.state_covs, R=0.0, H=self.H)
        return torch.diag_embed(means)

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        # TODO: use monte-carlo instead.
        # aside from the problem of ignoring state cov when means<POISSON_SMALL_THRESH, the other problem is
        # that means is itself an estimate, so even when means > POISSON_SMALL_THRESH, true value might be less
        if means.shape[-1] > 1:
            raise NotImplementedError("log-prob not currently implemented for poisson when there are multiple measures")
        use_mvnorm = (means >= POISSON_SMALL_THRESH).any(-1)
        out = torch.zeros_like(obs[..., 0])
        # use normal approximation for larger values:
        out[use_mvnorm] = torch.distributions.MultivariateNormal(
            loc=means[use_mvnorm],
            covariance_matrix=covs[use_mvnorm],
            validate_args=False
        ).log_prob(obs[use_mvnorm])
        # ignore state-cov and use poisson for smaller values:
        out[~use_mvnorm] = self.distribution_cls(
            means[~use_mvnorm].squeeze(-1), validate_args=False
        ).log_prob(obs[~use_mvnorm].squeeze(-1))
        return out

    def sample(self) -> Tensor:
        raise NotImplementedError

    @classmethod
    def _get_quantiles(cls, mean, std, conf: float, observed: bool) -> tuple:
        if not observed:
            return super()._get_quantiles(mean=mean, std=std, conf=conf, observed=observed)
        assert conf >= .50
        lower_conf = (1 - conf) / 2
        upper_conf = 1 - lower_conf
        lower = scipy_special.pdtrik(lower_conf, mean)
        upper = scipy_special.pdtrik(upper_conf, mean)
        return lower, upper
