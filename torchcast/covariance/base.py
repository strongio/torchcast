import math

from typing import List, Iterable, Optional, Tuple, Sequence
from warnings import warn

import torch

from torch import Tensor, nn, jit
from typing_extensions import Final

from torchcast.covariance.util import num_off_diag
from torchcast.internals.utils import get_owned_kwarg, is_near_zero
from torchcast.process.base import Process


class Covariance(nn.Module):
    """
    The :class:`.Covariance` can be used when you'd like more control over the covariance specification of a state-
    space model. For example, if you're training on diverse time-serieses that vary in scale/behavior, you could use an
    :class:`torch.nn.Embedding` to predict the variance of each series, with the group-ids as predictors:

    .. code-block:: python3

        kf = KalmanFilter(
            measures=measures,
            processes=processes,
            measure_covariance=Covariance.from_measures(
                measures,
                predict_variance=torch.nn.Embedding(len(group_ids), len(measures), padding_idx=0)
            ),
            process_covariance=Covariance.from_processes(
                processes,
                predict_variance=torch.nn.Embedding(
                    len(group_ids),
                    Covariance.from_processes(processes).param_rank,
                    padding_idx=0
                )
            )
        )

    """
    var_predict_multi: Final[float] = 0.1
    """
    :cvar var_predict_multi: If ``predict_variance`` are standard modules like :class:`torch.nn.Linear` or
     :class:`torch.nn.Embedding`, the random inits can often result in extreme variance-multipliers; these poor
     inits can make early optimization unstable. `var_predict_multi` (default 0.1) simply multiplies the output
     of ``predict_variance`` before passing them though :func:`torch.exp`; this serves to dampen initial outputs
     while still allowing large predictions if these are eventually warranted.
    """

    @classmethod
    def from_processes(cls,
                       processes: Sequence[Process],
                       cov_type: str = 'process',
                       predict_variance: Optional[nn.Module] = None,
                       **kwargs) -> 'Covariance':
        """
        :param processes: The ``processes`` used in your :class:`.StateSpaceModel`.
        :param cov_type: The type of covariance, either 'process' or 'initial' (default: 'process').
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, self.param_rank)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.KalmanFilter`.
        """

        assert cov_type in {'process', 'initial'}
        state_rank = 0
        no_cov_idx = []
        for p in processes:
            no_cov_elements = []
            if cov_type == 'process':
                no_cov_elements = p.fixed_state_elements or []
            for i, se in enumerate(p.state_elements):
                if se in no_cov_elements:
                    no_cov_idx.append(state_rank + i)
            state_rank += len(p.state_elements)

        if cov_type == 'process':
            # by default, assume process cov is less than measure cov:
            if 'init_diag_multi' not in kwargs:
                kwargs['init_diag_multi'] = .01
            if 'method' in kwargs and kwargs['method'] == 'low_rank':
                warn("``method='low_rank'`` not recommended for processes, consider 'low_rank+block_diag'")
        elif cov_type == 'initial':
            if (state_rank - len(no_cov_idx)) >= 10:
                # by default, use low-rank parameterization for initial cov:
                if 'method' not in kwargs:
                    kwargs['method'] = 'low_rank'
        else:
            raise ValueError(f"Unrecognized cov_type {cov_type}, expected 'initial' or 'process'.")

        return cls(
            rank=state_rank,
            empty_idx=no_cov_idx,
            id=f'{cov_type}_covariance',
            predict_variance=predict_variance,
            **kwargs
        )

    @classmethod
    def from_measures(cls,
                      measures: Sequence[str],
                      predict_variance: Optional[nn.Module] = None,
                      **kwargs) -> 'Covariance':
        """
        :param measures: The ``measures`` used in your :class:`.KalmanFilter`.
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, num_measures)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param kwargs: Other arguments passed to :func:`Covariance.__init__`.
        :return: A :class:`.Covariance` object that can be used in your :class:`.KalmanFilter`.
        """
        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        if 'method' not in kwargs and len(measures) > 5:
            kwargs['method'] = 'low_rank'
        if 'init_diag_multi' not in kwargs:
            kwargs['init_diag_multi'] = 1.0
        return cls(rank=len(measures), id='measure_covariance', predict_variance=predict_variance, **kwargs)

    def __init__(self,
                 rank: int,
                 method: str = 'log_cholesky',
                 empty_idx: List[int] = (),
                 predict_variance: Optional[nn.Module] = None,
                 id: Optional[str] = None,
                 init_diag_multi: float = 0.1):
        """
        You should rarely call this directly. Instead, call :func:`Covariance.from_measures` and
        :func:`Covariance.from_processes`.

        :param rank: The number of elements along the diagonal.
        :param method: The parameterization for the covariance. The default, "log_cholesky", parameterizes the
         covariance using the cholesky factorization (which is itself split into two tensors: the log-transformed
         diagonal elements and the off-diagonal). The other currently supported option is "low_rank", which
         parameterizes the covariance with two tensors: (a) the log-transformed std-deviations, and (b) a 'low rank'
         G*K tensor where G is the number of random-effects and K is int(sqrt(G)). Then the covariance is
         ``D + V @ V.t()`` where D is a diagonal-matrix with the std-deviations**2, and V is the low-rank tensor.
        :param empty_idx: In some cases (e.g. process-covariance) we will have some elements with no variance.
        :param predict_variance: A :class:`torch.nn.Module` that will predict a (log) multiplier for the variance.
         These should output real-values with shape ``(num_groups, self.param_rank)``; these values will then be
         converted to multipliers by applying :func:`torch.exp` (i.e. don't pass the output through a softplus).
        :param id: Identifier for this covariance. Typically left ``None`` and set when passed to the
         :class:`.KalmanFilter`.
        :param init_diag_multi: A float that will be applied as a multiplier to the initial values along the diagonal.
         This can be useful to provide intelligent starting-values to speed up optimization.
        """

        super(Covariance, self).__init__()

        self.id = id
        self.rank = rank

        empty_idx = set(empty_idx)
        assert all(isinstance(x, int) for x in empty_idx)
        self.param_rank = self.rank - len(empty_idx)
        mask = torch.zeros((self.rank, self.param_rank))
        c = 0
        for r in range(self.rank):
            if r not in empty_idx:
                mask[r, c] = 1.
                c += 1
        self.register_buffer('mask', mask)

        self._set_params(method, init_diag_multi)

        self.var_predict_module = predict_variance

        # TODO: allow user-defined?
        self.expected_kwarg = '' if self.var_predict_module is None else 'X'

    def _set_params(self, method: str, init_diag_multi: float):
        self.cholesky_log_diag: Optional[nn.Parameter] = None
        self.cholesky_off_diag: Optional[nn.Parameter] = None
        self.lr_mat: Optional[nn.Parameter] = None
        self.log_std_devs: Optional[nn.Parameter] = None
        if method == 'log_cholesky':
            self.method = method
            self.cholesky_log_diag = nn.Parameter(.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
            self.cholesky_off_diag = nn.Parameter(.1 * torch.randn(num_off_diag(self.param_rank)))
        elif method.startswith('low_rank'):
            self.method = 'low_rank'
            low_rank = method.replace('low_rank', '')
            if low_rank:
                low_rank = int(low_rank)
            else:
                low_rank = int(math.sqrt(self.param_rank))
            self.lr_mat = nn.Parameter(data=.01 * torch.randn(self.param_rank, low_rank))
            self.log_std_devs = nn.Parameter(data=.1 * torch.randn(self.param_rank) + math.log(init_diag_multi))
        else:
            raise NotImplementedError(method)

    @jit.ignore
    def set_id(self, id: str) -> 'Covariance':
        if self.id and id != self.id:
            warn(f"Id already set to {self.id}, overwriting")
        self.id = id
        return self

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, Tensor]]:
        if self.expected_kwarg:
            found_key, value = get_owned_kwarg(self.id, self.expected_kwarg, kwargs)
            yield found_key, self.expected_kwarg, torch.as_tensor(value)

    @staticmethod
    def log_chol_to_chol(log_diag: torch.Tensor, off_diag: torch.Tensor) -> torch.Tensor:
        assert log_diag.shape[:-1] == off_diag.shape[:-1]

        rank = log_diag.shape[-1]
        L1 = torch.diag_embed(torch.exp(log_diag))

        L2 = torch.zeros_like(L1)
        mask = torch.tril_indices(rank, rank, offset=-1)
        L2[mask[0], mask[1]] = off_diag
        return L1 + L2

    def _get_mini_cov(self) -> Tensor:
        if self.method == 'log_cholesky':
            assert self.cholesky_log_diag is not None
            assert self.cholesky_off_diag is not None
            L = self.log_chol_to_chol(self.cholesky_log_diag, self.cholesky_off_diag)
            mini_cov = L @ L.t()
        elif self.method == 'low_rank':
            assert self.lr_mat is not None
            assert self.log_std_devs is not None
            mini_cov = (
                    self.lr_mat @ self.lr_mat.t() +
                    torch.diag_embed(self.log_std_devs.exp() ** 2)
            )
        else:
            raise NotImplementedError(self.method)

        if is_near_zero(mini_cov.diagonal(dim1=-2, dim2=-1), atol=1e-12).any():
            warn(
                f"`{self.id}` has near-zero along the diagonal. Will add 1e-12 to the diagonal. "
                f"Values:\n{mini_cov.diag()}"
            )
            mini_cov = mini_cov + torch.eye(mini_cov.shape[-1], device=mini_cov.device, dtype=mini_cov.dtype) * 1e-12
        return mini_cov

    def forward(self, input: Optional[Tensor] = None, _ignore_input: bool = False) -> Tensor:
        # TODO: `multi_forward()` that only computes ``_get_mini_cov()`` once if we have time-varying inputs
        mini_cov = self._get_mini_cov()

        pred = None
        if self.var_predict_module is not None and not _ignore_input:
            pred = self.var_predict_multi * self.var_predict_module(input)
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                raise RuntimeError(f"{self.id}'s `predict_variance` produced nans/infs")
            if len(pred.shape) == 1:
                raise ValueError(
                    f"{self.id} `predict_variance` module output should have 2D output, got {len(pred.shape)}"
                )
            elif pred.shape[-1] not in (1, self.param_rank):
                raise ValueError(
                    f"{self.id} `predict_variance` module output should have `shape[-1]` of "
                    f"{self.param_rank}, got {pred.shape[-1]}"
                )
        if pred is not None:
            diag_multi = torch.diag_embed(torch.exp(pred))
            mini_cov = diag_multi @ mini_cov @ diag_multi

        return self.mask @ mini_cov @ self.mask.t()
        # return pad_covariance(mini_cov, [int(i not in self.empty_idx) for i in range(self.rank)])