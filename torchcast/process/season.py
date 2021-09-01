import copy
import itertools
import math
from typing import Optional, Tuple, Iterable, Sequence, Union
from warnings import warn

import numpy as np

import torch
from torch import jit, nn, Tensor

from torchcast.process.base import Process
from torchcast.process.utils import SingleOutput, Multi, Bounded, TimesToFourier, ScriptSequential


class _Season:

    @staticmethod
    def _standardize_period(period: Union[str, np.timedelta64], dt_unit_ns: Optional[float]) -> float:
        if dt_unit_ns is None:
            if not isinstance(period, (float, int)):
                raise ValueError(f"period is {type(period)}, but expected float/int since dt_unit is None.")
        else:
            if not isinstance(period, (float, int)):
                if isinstance(period, str):
                    period = np.timedelta64(1, period)
                period = period / (dt_unit_ns * np.timedelta64(1, 'ns'))
        return float(period)

    @staticmethod
    def _get_dt_unit_ns(dt_unit_str: str) -> int:
        if isinstance(dt_unit_str, np.timedelta64):
            dt_unit = dt_unit_str
        else:
            dt_unit = np.timedelta64(1, dt_unit_str)
        dt_unit_ns = dt_unit / np.timedelta64(1, 'ns')
        assert dt_unit_ns.is_integer()
        return int(dt_unit_ns)

    @jit.ignore
    def _standardize_offsets(self, offsets: Sequence) -> np.ndarray:
        if self.dt_unit_ns is None:
            return np.asanyarray(offsets) % self.period
        offsets = np.asanyarray(offsets, dtype='datetime64[ns]')
        ns_since_epoch = (offsets - np.datetime64(0, 'ns')).view('int64')
        offsets = ns_since_epoch % (self.period * self.dt_unit_ns) / self.dt_unit_ns  # todo: cancels out?
        return offsets


class Season(_Season, Process):
    """
    Method from `De Livera, A.M., Hyndman, R.J., & Snyder, R. D. (2011)`; in that paper TBATS refers to
    the whole model; here it is specifically the novel approach to modeling seasonality that they proposed.

    :param id: Unique identifier for this process.
    :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
     used in the kalman-filter -- i.e., how far we advance with every timestep. Can be `None` if the data are in
     arbitrary (non-datetime) units.
    :param period: The number of timesteps it takes to get through a full seasonal cycle. Does not have to be an
     integer (e.g. 365.25 for yearly to account for leap-years). Can also be a numpy.timedelta64 (or string that
     will be converted to one).
    :param K: The number of the fourier components.
    :param measure: The name of the measure for this process.
    :param fixed: Whether the seasonal-structure is allowed to evolve over time, or is fixed (default:
     ``fixed=False``). Setting this to ``True`` can be helpful for limiting the uncertainty of long-range forecasts.
    :param decay: By default, the seasonal structure will remain as the forecast horizon increases. An alternative is
     to allow this structure to decay (i.e. pass ``True``). If you'd like more fine-grained control over this decay,
     you can specify the min/max decay as a tuple (passing ``True`` uses a default value of ``(.98, 1.0)``).
    """

    def __init__(self,
                 id: str,
                 period: Union[float, str],
                 dt_unit: Optional[str],
                 K: int,
                 measure: Optional[str] = None,
                 fixed: bool = False,
                 decay: Optional[Tuple[float, float]] = None):

        self.dt_unit_ns = None if dt_unit is None else self._get_dt_unit_ns(dt_unit)
        self.period = self._standardize_period(period, self.dt_unit_ns)
        if self.period.is_integer() and self.period < K * 2:
            warn(f"K is larger than necessary given a period of {self.period}.")

        if isinstance(decay, bool) and decay:
            decay = (.98, 1.00)
        if isinstance(decay, tuple) and (decay[0] ** self.period) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )

        state_elements, transitions, h_tensor = self._setup(K=K, period=self.period, decay=decay)

        super().__init__(
            id=id,
            state_elements=state_elements,
            measure=measure,
            fixed_state_elements=state_elements if fixed else [],
        )
        if not decay:
            self.f_tensors.update(transitions)
        else:
            self.f_modules.update(transitions)

        self.h_tensor = torch.tensor(h_tensor)

    def _build_h_mat(self, inputs: dict, num_groups: int, num_times: int) -> Tensor:
        return self.h_tensor

    @staticmethod
    def _setup(K: int,
               period: float,
               decay: Optional[Tuple[float, float]]) -> Tuple[Sequence[str], dict, Sequence[float]]:

        if isinstance(decay, nn.Module):
            decay = [copy.deepcopy(decay) for _ in range(K * 2)]
        if isinstance(decay, (list, tuple)):
            are_modules = [isinstance(m, nn.Module) for m in decay]
            if any(are_modules):
                assert all(are_modules), "`decay` is a list with some modules on some other types"
                assert len(decay) == K * 2, "`decay` is a list of modules, but its length != K*2"
            else:
                assert len(decay) == 2, "if `decay` is not list of modules, should be (float,float)"
                decay = [SingleOutput(transform=Bounded(*decay)) for _ in range(K * 2)]

        state_elements = []
        f_tensors = {}
        f_modules = {}
        h_tensor = []
        for j in range(1, K + 1):
            sj = f"s{j}"
            state_elements.append(sj)
            h_tensor.append(1.)
            s_star_j = f"s*{j}"
            state_elements.append(s_star_j)
            h_tensor.append(0.)

            lam = torch.tensor(2. * math.pi * j / period)

            f_tensors[f'{sj}->{sj}'] = torch.cos(lam)
            f_tensors[f'{sj}->{s_star_j}'] = -torch.sin(lam)
            f_tensors[f'{s_star_j}->{sj}'] = torch.sin(lam)
            f_tensors[f'{s_star_j}->{s_star_j}'] = torch.cos(lam)

            if decay:
                # more complicated to support decay for TBATS b/c it already uses the transition matrix.
                # we'd like to keep the sj/starj transitions, just multiply them by the 0-1 decay
                for from_, to_ in itertools.product([sj, s_star_j], [sj, s_star_j]):
                    tkey = f'{from_}->{to_}'
                    which = 2 * (j - 1) + int(from_ == to_)
                    f_modules[tkey] = ScriptSequential([decay[which], Multi(f_tensors[tkey])])

        if f_modules:
            assert len(f_modules) == len(f_tensors)
            return state_elements, f_modules, h_tensor
        else:
            return state_elements, f_tensors, h_tensor

    @jit.ignore
    def offset_initial_state(self, initial_state: Tensor, start_offsets: Optional[Sequence] = None) -> Tensor:
        if start_offsets is None:
            if self.dt_unit_ns is None:
                return initial_state
            raise RuntimeError(f"Process '{self.id}' has `dt_unit`, so need to pass datetimes for `start_offsets`")

        start_offsets = self._standardize_offsets(start_offsets)
        # TODO: this is imprecise for non-integer periods
        start_offsets = start_offsets.round()
        num_groups = len(start_offsets)
        assert initial_state.shape[0] == num_groups

        F = self._build_f_mat({}, num_groups=num_groups, num_times=1)[:, 0]

        means = []
        mean = initial_state.unsqueeze(-1)
        for i in range(int(self.period) + 1):
            means.append(mean.squeeze(-1))
            mean = F @ mean

        groups = [i for i in range(num_groups)]
        times = [int(start_offsets[i].item()) for i in groups]
        return torch.stack(means, 1)[(groups, times)]


TBATS = Season


class FourierSeason(_Season):
    """
    A process which captures seasonal patterns using fourier serieses. Essentially a ``LinearModel`` where the
    model-matrix construction is done for you. Only recommended for toy problems; Consider ``TBATS`` or a
    ``LinearModel`` w/``torchcast.utils.add_season_features``.
    """

    def __init__(self,
                 id: str,
                 dt_unit: Optional[str],
                 period: Union[float, str],
                 K: int,
                 measure: Optional[str] = None,
                 fixed: bool = True,
                 decay: Optional[Union[nn.Module, Tuple[float, float]]] = None):
        """
        :param id: A unique identifier for this process
        :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
        used in the kalman-filter -- i.e., how far we advance with every timestep. Can be `None` if the data are in
        arbitrary (non-datetime) units.
        :param period: The number of timesteps it takes to get through a full seasonal cycle. Does not have to be an
        integer (e.g. 365.25 for yearly to account for leap-years). Can also be a numpy.timedelta64 (or string that
        will be converted to one).
        :param K: The number of the fourier components
        :param measure: The name of the measure for this process.
        :param fixed: TODO
        :param decay: TODO
        """
        raise NotImplementedError("TODO")
        warn(
            "This process is only recommended for toy problems. "
            "Consider `TBATS` or a `LinearModel` w/`torchcast.utils.add_season_features`",
        )

        self.dt_unit_ns: Optional[int] = None if dt_unit is None else self._get_dt_unit_ns(dt_unit)
        self.period = self._standardize_period(period, self.dt_unit_ns)

        if self.period.is_integer() and self.period < K * 2:
            warn(f"K is larger than necessary given a period of {self.period}.")

        if isinstance(decay, tuple) and (decay[0] ** self.period) < .01:
            warn(
                f"Given the seasonal period, the lower bound on `{id}`'s `decay` ({decay}) may be too low to "
                f"generate useful gradient information for optimization."
            )

        state_elements = []
        for j in range(K):
            state_elements.append(f'sin{j}')
            state_elements.append(f'cos{j}')

        super().__init__(
            id=id,
            predictors=state_elements,
            measure=measure,
            h_module=TimesToFourier(K=K, seasonal_period=float(self.period)),
            fixed=fixed,
            decay=decay
        )
        self.h_kwarg = 'current_times'

    @jit.ignore
    def get_kwargs(self, kwargs: dict) -> Iterable[Tuple[str, str, str, Tensor]]:
        offsets = self._standardize_offsets(kwargs['start_datetimes'])
        offsets = torch.as_tensor(offsets).view(-1, 1, 1)
        kwargs['current_times'] = offsets + kwargs['current_timestep']
        for found_key, key_name, value in Process.get_kwargs(self, kwargs):
            if found_key == 'current_times' and self.dt_unit_ns is not None:
                found_key = 'start_datetimes'
            yield found_key, key_name, value
