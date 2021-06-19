from typing import Sequence, Optional, Tuple, List, Dict, Iterable

import torch
from torch import Tensor

from torchcast.state_space import Predictions
from torchcast.exp_smooth.innovation_matrix import InnovationMatrix
from torchcast.covariance import Covariance
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep


class ExpSmoothStep(StateSpaceStep):

    def _mask_mats(self,
                   groups: Tensor,
                   val_idx: Optional[Tensor],
                   input: Tensor,
                   kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, Tensor]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        if val_idx is None:
            return input[groups], {k: v[groups] for k, v in kwargs.items()}
        else:
            m1d = torch.meshgrid(groups, val_idx)
            m2d = torch.meshgrid(groups, val_idx, val_idx)
            masked_input = input[m1d[0], m1d[1]]
            masked_kwargs = {
                'H': kwargs['H'][m1d[0], m1d[1]],
                'R': kwargs['R'][m2d[0], m2d[1], m2d[2]],
                'K': kwargs['K'][m1d[0], m1d[1]],
            }
            return masked_input, masked_kwargs

    def _update(self,
                input: Tensor,
                mean: Tensor,
                cov: Tensor,
                kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        measured_mean = (kwargs['H'] @ mean.unsqueeze(-1)).squeeze(-1)
        resid = input - measured_mean
        new_mean = mean + (kwargs['K'] @ resid.unsqueeze(-1)).squeeze(-1)
        new_cov = torch.zeros_like(cov)
        return new_mean, new_cov

    def predict(self, mean: Tensor, cov: Tensor, kwargs: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        F = kwargs['F']
        K = kwargs['K']
        R = kwargs['R']
        mean = (F @ mean.unsqueeze(-1)).squeeze(-1)
        # TODO: cheaper to check cov!=0 before applying FCF'?
        cov = F @ cov @ F.permute(0, 2, 1) + K @ R @ K.permute(0, 2, 1)
        return mean, cov


class ExpSmooth(StateSpaceModel):
    """
    Uses exponential smoothing to generate forecasts.
    """
    ss_step_cls = ExpSmoothStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 measure_covariance: Optional[Covariance] = None,
                 predict_smoothing: Optional[torch.nn.Module] = None):

        if measure_covariance is None:
            measure_covariance = Covariance.from_measures(measures)

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
        )

        state_rank = 0
        fixed_idx = []
        for p in processes:
            midx = self.measures.index(p.measure)
            fixed_els = p.fixed_state_elements or []
            for i, se in enumerate(p.state_elements):
                if se in fixed_els:
                    fixed_idx.append((state_rank + i, midx))
                # TODO: warn if time-varying H but some unfixed_els
            state_rank += len(p.state_elements)

        self.innovation_matrix = InnovationMatrix(
            measure_rank=len(self.measures),
            state_rank=state_rank,
            empty_idx=fixed_idx,
            predict_module=predict_smoothing
        ).set_id('innovation_matrix')

    def initial_covariance(self, *args, **kwargs) -> Tensor:
        ms = self._get_measure_scaling()
        return torch.eye(self.state_rank, dtype=ms.dtype, device=ms.device)

    @torch.jit.ignore()
    def design_modules(self) -> Iterable[Tuple[str, torch.nn.Module]]:
        # torchscript doesn't support super, see: https://github.com/pytorch/pytorch/issues/42885
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'measure_covariance', self.measure_covariance
        yield 'innovation_matrix', self.innovation_matrix

    @torch.jit.ignore
    def _generate_predictions(self,
                              means: Tensor,
                              covs: Tensor,
                              predict_kwargs: Dict[str, List[Tensor]],
                              update_kwargs: Dict[str, List[Tensor]]) -> 'Predictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """
        return Predictions(
            state_means=means,
            state_covs=covs,
            R=torch.stack(predict_kwargs['R'], 1),
            H=torch.stack(update_kwargs['H'], 1),
            kalman_filter=self
        )

    def build_design_mats(self,
                          static_kwargs: Dict[str, Dict[str, Tensor]],
                          time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                          num_groups: int,
                          out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        Fs, Hs = self._build_transition_and_measure_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)

        # measure-variance:
        Rs = self._build_measure_var_mats(static_kwargs, time_varying_kwargs, num_groups, out_timesteps)

        # innovation matrix / kalman-gain:
        if 'innovation_matrix' in time_varying_kwargs and \
                self.innovation_matrix.expected_kwarg in time_varying_kwargs['innovation_matrix']:
            pvar_inputs = time_varying_kwargs['innovation_matrix'][self.innovation_matrix.expected_kwarg]
            Ks: List[Tensor] = []
            for t in range(out_timesteps):
                Ks.append(self.innovation_matrix(pvar_inputs[t]))
        else:
            pvar_input: Optional[Tensor] = None
            if 'innovation_matrix' in static_kwargs and \
                    self.innovation_matrix.expected_kwarg in static_kwargs['innovation_matrix']:
                pvar_input = static_kwargs['innovation_matrix'].get(self.innovation_matrix.expected_kwarg)
            K = self.innovation_matrix(pvar_input)
            if len(K.shape) == 2:
                K = K.expand(num_groups, -1, -1)
            Ks = [K] * out_timesteps

        predict_kwargs = {'F': Fs, 'K': Ks, 'R': Rs}
        update_kwargs = {'H': Hs, 'K': Ks}
        return predict_kwargs, update_kwargs
