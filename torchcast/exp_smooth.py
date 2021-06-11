from typing import Sequence, Optional, Type, Tuple, List, Dict

import torch
from torch import Tensor

from internals.utils import get_nan_groups
from state_space import Predictions
from torchcast.covariance import Covariance, DiagCovariance
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep


class ExpSmoothStep(StateSpaceStep):

    # this would ideally be a class-attribute but torch.jit.trace strips them
    @torch.jit.ignore()
    def get_distribution(self) -> Type[torch.distributions.Distribution]:
        return torch.distributions.MultivariateNormal

    # noinspection DuplicatedCode
    # (need duplication b/c torch.jit currently doesn't support super)
    def update(self, input: Tensor, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        assert len(input.shape) > 1
        if len(input.shape) != 2:
            raise NotImplementedError

        num_groups = input.shape[0]
        if mean.shape[0] != num_groups:
            assert mean.shape[0] == 1
            mean = mean.expand(num_groups, -1)
        if cov.shape[0] != num_groups:
            assert cov.shape[0] == 1
            cov = cov.expand(num_groups, -1)

        H = kwargs['H']
        R = kwargs['R']
        init_var = kwargs['init_var']

        isnan = torch.isnan(input)
        if isnan.all():
            return mean, cov
        if isnan.any():
            new_mean = mean.clone()
            new_cov = cov.clone()
            for groups, val_idx in get_nan_groups(isnan):
                if val_idx is None:
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[groups],
                        mean=mean[groups],
                        cov=cov[groups],
                        H=H[groups],
                        R=R[groups],
                        init_var=init_var[groups]
                    )
                else:
                    # masks:
                    m1d = torch.meshgrid(groups, val_idx)
                    new_mean[groups], new_cov[groups] = self._update(
                        input=input[m1d[0], m1d[1]],
                        mean=mean[groups],
                        cov=cov[groups],
                        H=H[m1d[0], m1d[1]],
                        R=R[m1d[0], m1d[1]],
                        init_var=init_var[m1d[0], m1d[1]]
                    )
            return new_mean, new_cov
        else:
            return self._update(input=input, mean=mean, cov=cov, H=H, R=R, init_var=init_var)

    # noinspection PyMethodOverriding
    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                H: Tensor,
                R: Tensor,
                init_var: Tensor) -> Tuple[Tensor, Tensor]:
        if R.shape[2] > 1:
            raise NotImplementedError("TODO")
        else:
            K = R.diag() / (cov + R.diag())
        measured_mean = (H @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (K @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = init_var
        return new_mean, new_cov

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        Q = kwargs['Q']
        mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        cov = cov + Q  # TODO: seems like a problem we're ignoring F?
        return mean, cov


class ExpSmooth(StateSpaceModel):
    ss_step_cls = ExpSmoothStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 process_covariance: Optional[Covariance] = None,
                 measure_covariance: Optional[Covariance] = None):

        if process_covariance is None:
            # TODO check class
            process_covariance = DiagCovariance.for_processes(processes, cov_type='process')

        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)

        initial_covariance = DiagCovariance.for_processes(processes, cov_type='initial')

        super().__init__(
            processes=processes,
            measures=measures,
            process_covariance=process_covariance,
            measure_covariance=measure_covariance,
            initial_covariance=initial_covariance
        )

        self.log_init_var_decay = torch.nn.Parameter(.01 * torch.randn(initial_covariance.rank))

    def _build_var_mats(self,
                        static_kwargs: Dict[str, Dict[str, Tensor]],
                        time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                        num_groups: int,
                        out_timesteps: int) -> Tuple[List[Tensor], List[Tensor]]:

        # process-variance:
        measure_scaling = self._get_measure_scaling()
        if 'process_covariance' in time_varying_kwargs and \
                self.process_covariance.expected_kwarg in time_varying_kwargs['process_covariance']:
            pvar_inputs = time_varying_kwargs['process_covariance'][self.process_covariance.expected_kwarg]
            Qs: List[Tensor] = []
            for t in range(out_timesteps):
                Qs.append(measure_scaling * self.process_covariance(pvar_inputs[t]))
        else:
            pvar_input: Optional[Tensor] = None
            if 'process_covariance' in static_kwargs and \
                    self.process_covariance.expected_kwarg in static_kwargs['process_covariance']:
                pvar_input = static_kwargs['process_covariance'].get(self.process_covariance.expected_kwarg)
            Q = measure_scaling * self.process_covariance(pvar_input)
            if len(Q.shape) == 2:
                Q = Q.expand(num_groups, -1, -1)
            Qs = [Q] * out_timesteps

        # measure-variance:
        if 'measure_covariance' in time_varying_kwargs and \
                self.measure_covariance.expected_kwarg in time_varying_kwargs['measure_covariance']:
            mvar_inputs = time_varying_kwargs['measure_covariance'][self.measure_covariance.expected_kwarg]
            Rs: List[Tensor] = []
            for t in range(out_timesteps):
                Rs.append(self.measure_covariance(mvar_inputs[t]))
        else:
            mvar_input: Optional[Tensor] = None
            if 'measure_covariance' in static_kwargs and \
                    self.measure_covariance.expected_kwarg in static_kwargs['measure_covariance']:
                mvar_input = static_kwargs['measure_covariance'].get(self.measure_covariance.expected_kwarg)
            R = self.measure_covariance(mvar_input)
            if len(R.shape) == 2:
                R = R.expand(num_groups, -1, -1)
            Rs = [R] * out_timesteps

        return Rs, Qs

    # noinspection DuplicatedCode
    def build_design_mats(self,
                          static_kwargs: Dict[str, Dict[str, Tensor]],
                          time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                          num_groups: int,
                          out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        # currently need to duplicate code b/c torch.jit doesn't support super()

        Rs, Qs = self._build_var_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)
        Fs, Hs = self._build_transition_and_measure_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)

        # TODO: this approach ignores the `initial_state` cov
        init_var_mats = []
        for t in range(out_timesteps):
            init_var_mats.append(self.initial_covariance() * 1 / t * torch.exp(self.log_init_var_decay))

        predict_kwargs = {'F': Fs, 'Q': Qs}
        update_kwargs = {'H': Hs, 'R': Rs, 'init_var': init_var_mats}
        return predict_kwargs, update_kwargs

    def _generate_predictions(self, means: Tensor, covs: Tensor, R: Tensor, H: Tensor) -> 'Predictions':
        return Predictions(
            state_means=means, state_covs=covs, R=torch.diag_embed(R), H=torch.diag_embed(H), kalman_filter=self
        )
