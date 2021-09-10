from typing import Tuple, Sequence, Optional, Dict
from warnings import warn

import torch

from torchcast.internals.utils import validate_gt_shape
from torchcast.process.base import Process
from torchcast.process.utils import Bounded, SingleOutput


class LinearModel(Process):
    """
    A process which takes a model-matrix of predictors, and each state corresponds to the coefficient on each.

    :param id: Unique identifier for the process
    :param predictors: A sequence of strings with predictor-names.
    :param measure: The name of the measure for this process.
    :param fixed: By default, the regression-coefficients are assumed to be fixed: we are initially
     uncertain about their value at the start of each series, but we gradually grow more confident. If
     ``fixed=False`` then we continue to inject uncertainty at each timestep so that uncertainty asymptotes
     at some nonzero value. This amounts to dynamic-regression where the coefficients evolve over-time. Note only
     ``KalmanFilter`` (but not ``ExpSmoother``) supports this.
    :param decay: By default, the coefficient-values will remain as the forecast horizon increases. An alternative is
     to allow these to decay (i.e. pass ``True``). If you'd like more fine-grained control over this decay,
     you can specify the min/max decay as a tuple (passing ``True`` uses a default value of ``(.98, 1.0)``).
    """

    def __init__(self,
                 id: str,
                 predictors: Sequence[str],
                 measure: Optional[str] = None,
                 fixed: bool = True,
                 decay: Optional[Tuple[float, float]] = None):

        super().__init__(
            id=id,
            state_elements=predictors,
            measure=measure,
            fixed_state_elements=predictors if fixed else []
        )

        if decay is None:
            self.f_tensors['all_self'] = torch.ones(len(predictors))
        else:
            if fixed:
                warn("decay=True, fixed=True not recommended.")
            if decay is True:
                decay = (.98, 1.0)
            if isinstance(decay, tuple):
                decay = SingleOutput(numel=len(predictors), transform=Bounded(*decay))
            self.f_modules['all_self'] = decay
        self.expected_kwargs = ['X']

    def _build_h_mat(self, inputs: Dict[str, torch.Tensor], num_groups: int, num_times: int) -> torch.Tensor:
        # if not torch.jit.is_scripting():
        #     try:
        #         X = inputs['X']
        #     except KeyError as e:
        #         raise TypeError(f"Missing required keyword-arg `X` (or `{self.id}__X`).") from e
        # else:
        X = inputs['X']
        assert not torch.isnan(X).any()
        assert not torch.isinf(X).any()

        X = validate_gt_shape(X, num_groups, num_times, trailing_dim=[self.rank])
        # note: trailing_dim is really (self.rank, self.measures), but currently processes can only have one measure

        return X
