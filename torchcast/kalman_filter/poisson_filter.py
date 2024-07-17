"""
The :class:`.PoissonFilter` is a :class:`torch.nn.Module` which generates forecasts using the extended kalman-filter.
It uses a softplus link function to link the state-mean to the observed mean, and fixes the measure-variance to equal
the mean.

This class is experimental, and the log-likelihood and predictions do not currently incorporate the state-covariance.

----------
"""
from typing import Dict, Tuple, Optional, Type, Sequence, Iterable, List, TYPE_CHECKING
from warnings import warn

import numpy as np
import torch
from scipy import special as scipy_special
from torch import Tensor, nn

from functools import cached_property
from torch.nn import Softplus

from .ekf import EKFStep
from ..covariance import Covariance
from ..process import Process
from ..state_space import StateSpaceModel, Predictions

if TYPE_CHECKING:
    from pandas import DataFrame

POISSON_SMALL_THRESH = 10

softplus = Softplus()


def inverse_softplus(x: torch.Tensor, eps: float = .001) -> torch.Tensor:
    not_too_big = x < 20
    out = x.clone()
    x = x.clamp(min=eps)
    out[not_too_big] = torch.log(torch.exp(x[not_too_big]) - 1)
    return out


class PoissonStep(EKFStep):
    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.Poisson

    def _adjust_h(self, mean: Tensor, H: Tensor) -> Tensor:
        raise NotImplementedError("TODO")

    def _adjust_r(self, measured_mean: Tensor, R: Tensor) -> Tensor:
        return torch.diag_embed(measured_mean)  # variance = mean

    def _link(self, measured_mean: Tensor) -> Tensor:
        return softplus(measured_mean)


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
        raise NotImplementedError("TODO")

    @classmethod
    def plot(cls,
             df: 'DataFrame',
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> 'DataFrame':
        raise NotImplementedError("TODO")

    @cached_property
    def R(self) -> torch.Tensor:
        raise NotImplementedError("TODO")

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        raise NotImplementedError("TODO")

    def sample(self) -> Tensor:
        raise NotImplementedError("TODO")

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
