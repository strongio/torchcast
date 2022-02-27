from typing import Tuple, Sequence, Optional, Dict
from warnings import warn

import torch

from torchcast.internals.utils import validate_gt_shape
from torchcast.process.base import Process
from torchcast.process.utils import Bounded, SingleOutput
from torchcast.utils.data import chunk_grouped_data


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

    @staticmethod
    def _l2_solve(y: torch.Tensor,
                  X: torch.Tensor,
                  prior_precision: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Given tensors y,X in the format expected by ``StateSpaceModel`` -- 3D arrays with groups*times*measures --
        return the solutions to a linear-model for each group.

        See :func:`~torchcast.process.LinearModel.solve_and_predict()`

        :param y: A 3d tensor of time-series values.
        :param X: A 3d tensor of predictors.
        :param prior_precision: Optional. A penalty matrix.
        :return: The solution.
        """

        # handling nans requires flattening + scatter_add:
        num_groups, num_times, num_preds = X.shape
        X = X.view(-1, X.shape[-1])
        y = y.view(-1, y.shape[-1])
        is_valid = ~torch.isnan(y).squeeze()
        group_ids_broad = torch.repeat_interleave(torch.arange(num_groups, device=y.device), num_times)
        X = X[is_valid]
        y = y[is_valid]
        group_ids_broad = group_ids_broad[is_valid]

        # Xty:
        Xty_els = X * y
        Xty = torch.zeros(num_groups, num_preds, dtype=y.dtype, device=y.device). \
            scatter_add(0, group_ids_broad.unsqueeze(-1).expand_as(Xty_els), Xty_els)

        # XtX:
        XtX = torch.stack([Xg.t() @ Xg for Xg, in chunk_grouped_data(X, group_ids=group_ids_broad.cpu())])
        XtXp = XtX
        if prior_precision is not None:
            XtXp = XtXp + prior_precision

        return torch.linalg.solve(XtXp, Xty.unsqueeze(-1))

    @classmethod
    def solve_and_predict(cls,
                          y: torch.Tensor,
                          X: torch.Tensor,
                          prior_precision: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Given tensors y,X in the format expected by ``StateSpaceModel`` -- 3D arrays with groups*times*measures --
        solve the linear-model for each group, then generate predictions from these.

        This can be useful for pretraining a ``StateSpaceModel`` which uses the ``LinearModel`` class.

        :param y: A 3d tensor of time-series values.
        :param X: A 3d tensor of predictors.
        :param prior_precision: Optional. A penalty matrix.
        :return: A tensor with the same dimensions as ``y`` with the predictions.
        """
        coefs = cls._l2_solve(y=y, X=X, prior_precision=prior_precision)
        return (coefs.transpose(-1, -2) * X).sum(-1).unsqueeze(-1)

