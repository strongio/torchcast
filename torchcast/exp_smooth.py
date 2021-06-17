from typing import Sequence, Optional, Type, Tuple, List, Dict

import torch
from torch import Tensor

from torchcast.internals.utils import get_nan_groups
from torchcast.state_space import Predictions
from torchcast.covariance import Covariance, DiagCovariance
from torchcast.process import Process
from torchcast.state_space import StateSpaceModel
from torchcast.state_space.ss_step import StateSpaceStep



class ExpSmooth(StateSpaceModel):
    ss_step_cls = ExpSmoothStep

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 innovation_matrix: Optional['InnovationMatrix'] = None,
                 measure_covariance: Optional[Covariance] = None):

        if measure_covariance is None:
            measure_covariance = Covariance.for_measures(measures)
        if innovation_matrix is None:
            raise NotImplementedError("TODO")

        super().__init__(
            processes=processes,
            measures=measures,
            measure_covariance=measure_covariance,
        )

    def _prepare_initial_state(self, initial_state, start_offsets: Optional[Sequence] = None,
                               num_groups: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError("TODO")

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

        raise NotImplementedError("TODO")
        predict_kwargs = {'F': Fs}
        update_kwargs = {'H': Hs, 'R': Rs}
        return predict_kwargs, update_kwargs

    def _generate_predictions(self, means: Tensor, covs: Tensor, R: Tensor, H: Tensor) -> 'Predictions':
        return Predictions(
            state_means=means, state_covs=covs, R=torch.diag_embed(R), H=torch.diag_embed(H), kalman_filter=self
        )
