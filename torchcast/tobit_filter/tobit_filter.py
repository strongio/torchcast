from torchcast.tobit_filter.ss_step import CensoredGaussianStep
from torchcast.kalman_filter import KalmanFilter


class TobitFilter(KalmanFilter):
    ss_step_cls = CensoredGaussianStep

    def build_design_mats(self,
                          static_kwargs: Dict[str, Dict[str, Tensor]],
                          time_varying_kwargs: Dict[str, Dict[str, List[Tensor]]],
                          num_groups: int,
                          out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        raise NotImplementedError("pull lower/upper from time_varying_kwargs, re-add at end as update_kwargs")

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        raise NotImplementedError("pull lower/upper; always time-varying")

    # what else? need a different predictions for diff log prob? and need to pass lower/upper to it
