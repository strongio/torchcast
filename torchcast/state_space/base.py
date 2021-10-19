from collections import defaultdict
from typing import Tuple, List, Optional, Sequence, Dict, Iterable, Callable, Union, Type
from warnings import warn

import torch
from torch import nn, Tensor

from torchcast.internals.utils import get_owned_kwargs
from torchcast.covariance import Covariance
from torchcast.state_space.predictions import Predictions
from torchcast.state_space.ss_step import StateSpaceStep
from torchcast.process.regression import Process


class StateSpaceModel(nn.Module):
    """
    Base-class for any :class:`torch.nn.Module` which generates predictions/forecasts using a state-space model.

    :param processes: A list of :class:`.Process` modules.
    :param measures: A list of strings specifying the names of the dimensions of the time-series being measured.
    :param measure_covariance: A module created with ``Covariance.from_measures(measures)``.
    """
    ss_step_cls: Type[StateSpaceStep]

    def __init__(self,
                 processes: Sequence[Process],
                 measures: Optional[Sequence[str]] = None,
                 measure_covariance: Optional[Covariance] = None):
        super().__init__()

        if isinstance(measures, str):
            measures = [measures]
            warn(f"`measures` should be a list of strings not a string; interpreted as `{measures}`.")
        self._validate(processes, measures)

        self.measure_covariance = measure_covariance.set_id('measure_covariance')

        self.ss_step = self.ss_step_cls()

        # measures:
        self.measures = measures
        self.measure_to_idx = {m: i for i, m in enumerate(self.measures)}

        # processes:
        self.processes = nn.ModuleDict()
        self.process_to_slice: Dict[str, Tuple[int, int]] = {}
        self.state_rank = 0
        for p in processes:
            assert p.measure, f"{p.id} does not have its `measure` set"
            self.processes[p.id] = p
            self.process_to_slice[p.id] = (self.state_rank, self.state_rank + len(p.state_elements))
            self.state_rank += len(p.state_elements)

        # the initial mean
        self.initial_mean = torch.nn.Parameter(.1 * torch.randn(self.state_rank))

        # can disable for debugging/tests:
        self._scale_by_measure_var = True

    @torch.jit.ignore()
    def fit(self,
            *args,
            tol: float = .001,
            patience: int = 3,
            max_iter: int = 200,
            optimizer: Optional[torch.optim.Optimizer] = None,
            verbose: int = 2,
            callbacks: Sequence[Callable] = (),
            loss_callback: Optional[Callable] = None,
            callable_kwargs: Optional[Dict[str, Callable]] = None,
            set_initial_values: bool = True,
            **kwargs):
        """
        A high-level interface for invoking the standard model-training boilerplate. This is helpful to common cases in
        which the number of parameters is moderate and the data fit in memory. For other cases you are encouraged to
        roll your own training loop.

        :param args: A tensor containing the batch of time-series(es), see :func:`KalmanFilter.forward()`.
        :param tol: Stopping tolerance.
        :param patience: Patience: if loss changes by less than `tol` for this many epochs, then training will be
         stopped.
        :param max_iter: The maximum number of iterations after which training will stop regardless of loss.
        :param optimizer: The optimizer to use. Default is to create an instance of :class:`torch.optim.LBFGS` with
         args ``(max_iter=10, line_search_fn='strong_wolfe', lr=.5)``.
        :param verbose: If True (default) will print the loss and epoch; for :class:`torch.optim.LBFGS` optimizer
         (the default) this progress bar will tick within each epoch to track the calls to forward.
        :param callbacks: A list of functions that will be called at the end of each epoch, which take the current
         epoch's loss value.
        :param loss_callback: A callback that takes the loss and returns a modified loss, called before each call to
         `backward()`. This can be used for example to add regularization.
        :param callable_kwargs: Some keyword-arguments to :func:`KalmanFilter.forward()` aren't static, but need to be
         recomputed every time. ``callable_kwargs`` is a dictionary where the keys are keyword-names and the values
         are no-argument functions that will be called each iteration to recompute the corresponding arguments. For
         example, ``callable_kwargs={'initial_state' : lambda: my_initial_state_nn(group_ids)``.
        :param set_initial_values: Default is to set ``initial_mean`` to sensible value given ``y``. This helps speed
         up training if the data are not centered. Set to ``False`` if you're resuming training from a previous
         ``fit()`` call.
        :param kwargs: Further keyword-arguments passed to :func:`KalmanFilter.forward()`.
        :return: This `KalmanFilter` instance.
        """

        # this may change in the future
        assert len(args) == 1
        y = args[0]

        if optimizer is None:
            optimizer = torch.optim.LBFGS([p for p in self.parameters() if p.requires_grad],
                                          max_iter=10, line_search_fn='strong_wolfe', lr=.5)

        if set_initial_values:
            self.set_initial_values(y)

        prog = None
        if verbose > 1:
            try:
                from tqdm.auto import tqdm
                if isinstance(optimizer, torch.optim.LBFGS):
                    prog = tqdm(total=optimizer.param_groups[0]['max_eval'])
                else:
                    prog = tqdm(total=1)
            except ImportError:
                warn("`progress=True` requires package `tqdm`.")

        epoch = 0

        callable_kwargs = callable_kwargs or {}

        def closure():
            optimizer.zero_grad()
            kwargs.update({k: v() for k, v in callable_kwargs.items()})
            pred = self(y, **kwargs)
            loss = -pred.log_prob(y).mean()
            if loss_callback:
                loss = loss_callback(loss)
            loss.backward()
            if prog:
                prog.update()
                prog.set_description(f'Epoch: {epoch}; Loss: {loss:.4f}')
            return loss

        prev_train_loss = float('inf')
        num_lower = 0
        for epoch in range(max_iter):
            try:
                if prog:
                    prog.reset()
                train_loss = optimizer.step(closure).item()
                for callback in callbacks:
                    callback(train_loss)
                if abs(train_loss - prev_train_loss) < tol:
                    num_lower += 1
                else:
                    num_lower = 0
                if num_lower == patience:
                    break
                prev_train_loss = train_loss
            except KeyboardInterrupt:
                break
            finally:
                optimizer.zero_grad(set_to_none=True)

        return self

    @torch.jit.ignore()
    def set_initial_values(self, y: Tensor):
        if 'initial_mean' not in self.state_dict():
            return

        assert len(self.measures) == y.shape[-1]

        hits = {m: [] for m in self.measures}
        for pid in self.processes:
            process = self.processes[pid]
            # have to use the name since `jit.script()` strips the class
            if (getattr(process, 'original_name', None) or type(process).__name__) in ('LocalLevel', 'LocalTrend'):
                if 'position->position' in (process.f_modules or {}):
                    continue
                assert process.measure

                hits[process.measure].append(process.id)
                measure_idx = list(self.measures).index(process.measure)
                with torch.no_grad():
                    t0 = y[:, 0, measure_idx]
                    self.state_dict()['initial_mean'][self.process_to_slice[pid][0]] = \
                        t0[~torch.isnan(t0) & ~torch.isinf(t0)].mean()

        for measure, procs in hits.items():
            if len(procs) > 1:
                warn(
                    f"For measure '{measure}', multiple processes ({procs}) track the overall level; consider adding "
                    f"`decay` to all but one."
                )
            elif not len(procs):
                warn(
                    f"For measure '{measure}', no processes track the overall level; consider centering data in "
                    f"preprocessing prior to training (if you haven't already)."
                )

    @staticmethod
    def _validate(processes: Sequence[Process], measures: Sequence[str]):
        if not hasattr(measures, '__getitem__'):
            warn(f"`measures` appears to be an unordered collection -- needs to be ordered")

        assert len(measures) == len(set(measures))

        process_names = set()
        for p in processes:
            if p.id in process_names:
                raise ValueError(f"There are multiple processes with id '{p.id}'.")
            else:
                process_names.add(p.id)
            if isinstance(p, torch.jit.RecursiveScriptModule):
                raise TypeError(
                    f"Processes should not be wrapped in `torch.jit.script` *before* being passed to `KalmanFilter`"
                )
            if p.measure:
                if p.measure not in measures:
                    raise ValueError(f"'{p.id}' has measure '{p.measure}' not in `measures`.")
            else:
                if len(measures) > 1:
                    raise ValueError(f"Must set measure for '{p.id}' since there are multiple measures.")
                p.measure = measures[0]

    @torch.jit.ignore()
    def design_modules(self) -> Iterable[Tuple[str, nn.Module]]:
        for pid in self.processes:
            yield pid, self.processes[pid]
        yield 'measure_covariance', self.measure_covariance

    @torch.jit.ignore()
    def forward(self,
                *args,
                n_step: Union[int, float] = 1,
                start_offsets: Optional[Sequence] = None,
                out_timesteps: Optional[Union[int, float]] = None,
                initial_state: Union[Tensor, Tuple[Optional[Tensor], Optional[Tensor]]] = (None, None),
                every_step: bool = True,
                include_updates_in_output: bool = False,
                **kwargs) -> Predictions:
        """
        Generate n-step-ahead predictions from the model.

        :param args: A (group X time X measures) tensor. Optional if ``initial_state`` is specified.
        :param n_step: What is the horizon for the predictions output for each timepoint? Defaults to one-step-ahead
         predictions (i.e. n_step=1).
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``input``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
         array of datetimes here. Otherwise you can pass an array of integers. Or leave ``None`` if there are no
         seasonal processes.
        :param out_timesteps: The number of timesteps to produce in the output. This is useful when passing a tensor
         of predictors that goes later in time than the `input` tensor -- you can specify ``out_timesteps=X.shape[1]``
         to get forecasts into this later time horizon.
        :param initial_state: The initial prediction for the state of the system. XXX (single tensor for mean always
         supported. for kf child class, can pass (mean,cov) tuple. this is usually if you're feeding from a previous
         output. for exp-smooth child class, latter isn't supported.
        :param every_step: By default, ``n_step`` ahead predictions will be generated at every timestep. If
         ``every_step=False``, then these predictions will only be generated every `n_step` timesteps. For example,
         with hourly data, ``n_step=24`` and ``every_step=True``, each timepoint would be a forecast generated with
         data 24-hours in the past. But with ``every_step=False`` the first timestep would be 1-step-ahead, the 2nd
         would be 2-step-ahead, ... the 23rd would be 24-step-ahead, the 24th would be 1-step-ahead, etc. The advantage
         to ``every_step=False`` is speed: training data for long-range forecasts can be generated without requiring
         the model to produce and discard intermediate predictions every timestep.
        :param include_updates_in_output: If False, only the ``n_step`` ahead predictions are included in the output.
         This means that we cannot use this output to generate the ``initial_state`` for subsequent forward-passes. Set
         to True to allow this -- False by default to reduce memory.
        :param kwargs: Further arguments passed to the `processes`. For example, the :class:`.LinearModel` expects an
         ``X`` argument for predictors.
        :return: A :class:`.Predictions` object with :func:`Predictions.log_prob()` and
         :func:`Predictions.to_dataframe()` methods.
        """

        assert len(args) <= 1
        if len(args):
            input = args[0]
            assert torch.is_floating_point(input)
        else:
            input = None

        if out_timesteps is None and input is None:
            raise RuntimeError("If no input is passed, must specify `out_timesteps`")

        if isinstance(initial_state, Tensor):
            initial_state = (initial_state, None)
        initial_state = self._prepare_initial_state(
            initial_state,
            start_offsets=start_offsets,
        )

        if isinstance(n_step, float):
            if not n_step.is_integer():
                raise ValueError("`n_step` must be an int.")
            n_step = int(n_step)
        if isinstance(out_timesteps, float):
            if not out_timesteps.is_integer():
                raise ValueError("`out_timesteps` must be an int.")
            out_timesteps = int(out_timesteps)

        preds, updates, R, H = self._script_forward(
            input=input,
            initial_state=initial_state,
            n_step=n_step,
            every_step=every_step,
            out_timesteps=out_timesteps,
            kwargs_per_process=self._parse_design_kwargs(
                input,
                out_timesteps=out_timesteps or input.shape[1],
                **kwargs
            )
        )
        return self._generate_predictions(preds, R, H, updates if include_updates_in_output else None)

    @torch.jit.ignore
    def _generate_predictions(self,
                              preds: Tuple[Tensor, Tensor],
                              R: Tensor,
                              H: Tensor,
                              updates: Optional[Tuple[Tensor, Tensor]] = None) -> 'Predictions':
        """
        StateSpace subclasses may pass subclasses of `Predictions` (e.g. for custom log-prob)
        """

        kwargs = {
            'state_means': preds[0],
            'state_covs': preds[1],
            'R': R,
            'H': H,
            'model': self
        }
        if updates is not None:
            kwargs.update(update_means=updates[0], update_covs=updates[1])
        return Predictions(**kwargs)

    @torch.jit.ignore
    def _prepare_initial_state(self,
                               initial_state: Tuple[Optional[Tensor], Optional[Tensor]],
                               start_offsets: Optional[Sequence] = None) -> Tuple[Tensor, Tensor]:
        init_mean, init_cov = initial_state
        if init_mean is None:
            init_mean = self.initial_mean[None, :]
        elif len(init_mean.shape) != 2:
            raise ValueError(
                f"Expected ``init_mean`` to have two-dimensions for (num_groups, state_dim), got {init_mean.shape}"
            )

        if init_cov is None:
            init_cov = self.initial_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[:, 0]
        elif len(init_cov.shape) != 3:
            raise ValueError(
                f"Expected ``init_cov`` to be 3-D with (num_groups, state_dim, state_dim), got {init_cov.shape}"
            )

        measure_scaling = torch.diag_embed(self._get_measure_scaling().unsqueeze(0))
        init_cov = measure_scaling @ init_cov @ measure_scaling

        # seasonal processes need to offset the initial mean:
        if start_offsets is not None:
            if init_mean.shape[0] == 1:
                init_mean = init_mean.expand(len(start_offsets), -1)
            elif init_mean.shape[0] != len(start_offsets):
                raise ValueError("Expected ``len(start_offets) == initial_state[0].shape[0]``")

            init_mean_w_offset = []
            for pid in self.processes:
                p = self.processes[pid]
                _process_slice = slice(*self.process_to_slice[pid])
                init_mean_w_offset.append(p.offset_initial_state(init_mean[:, _process_slice], start_offsets))
            init_mean_offset = torch.cat(init_mean_w_offset, 1)
        else:
            init_mean_offset = init_mean

        return init_mean_offset, init_cov

    @torch.jit.export
    def _script_forward(self,
                        input: Optional[Tensor],
                        kwargs_per_process: Dict[str, Dict[str, Tensor]],
                        initial_state: Tuple[Tensor, Tensor],
                        n_step: int = 1,
                        out_timesteps: Optional[int] = None,
                        every_step: bool = True
                        ) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor], Tensor, Tensor]:
        """
        :param input: A (group X time X measures) tensor. Optional if `initial_state` is specified.
        :param kwargs_per_process: Keyword-arguments to the Processes TODO
        :param initial_state: A (mean, cov) tuple to use as the initial state.
        :param n_step: What is the horizon for predictions? Defaults to one-step-ahead (i.e. n_step=1).
        :param out_timesteps: The number of timesteps in the output. Might be longer than input if forecasting.
        :param every_step: Experimental. When n_step>1, we can generate these n-step-ahead predictions at every
         timestep (e.g. 24-hour-ahead predictions every hour), in which case we'd save the 24-step-ahead prediction.
         Alternatively, we could generate 24-hour-ahead predictions at every 24th hour, in which case we'd save
         predictions 1-24. The former corresponds to every_step=True, the latter to every_step=False. If n_step=1
         (the default) then this option has no effect.
        :return: predictions (tuple of (means,covs)), updates (tuple of (means,covs)), R, H
        """
        assert n_step > 0

        meanu, covu = initial_state

        if input is None:
            if out_timesteps is None:
                raise RuntimeError("If `input` is None must pass `out_timesteps`")
            inputs = []

            num_groups = meanu.shape[0]
        else:
            if len(input.shape) != 3:
                raise ValueError(f"Expected len(input.shape) == 3 (group,time,measure)")
            if input.shape[-1] != len(self.measures):
                raise ValueError(f"Expected input.shape[-1] == {len(self.measures)} (len(self.measures))")

            num_groups = input.shape[0]
            if meanu.shape[0] == 1:
                meanu = meanu.expand(num_groups, -1)
            if covu.shape[0] == 1:
                covu = covu.expand(num_groups, -1, -1)

            inputs = input.unbind(1)
            if out_timesteps is None:
                out_timesteps = len(inputs)

        predict_kwargs, update_kwargs = self._build_design_mats(
            kwargs_per_process=kwargs_per_process,
            num_groups=num_groups,
            out_timesteps=out_timesteps
        )

        # first loop through to do predict -> update
        meanus: List[Tensor] = []
        covus: List[Tensor] = []
        mean1s: List[Tensor] = []
        cov1s: List[Tensor] = []
        for t in range(out_timesteps):
            mean1step, cov1step = self.ss_step.predict(
                meanu,
                covu,
                {k: v[t] for k, v in predict_kwargs.items()}
            )
            mean1s.append(mean1step)
            cov1s.append(cov1step)
            if t < len(inputs):
                meanu, covu = self.ss_step.update(
                    inputs[t],
                    mean1step,
                    cov1step,
                    {k: v[t] for k, v in update_kwargs.items()}
                )
            else:
                meanu, covu = mean1step, cov1step
            meanus.append(meanu)
            covus.append(covu)

        # 2nd loop to get n_step updates:
        # idx: Dict[int, int] = {}
        meanps: Dict[int, Tensor] = {}
        covps: Dict[int, Tensor] = {}
        for t1 in range(out_timesteps):
            # tu: time of update
            # t1: time of 1step
            tu = t1 - 1

            # - if every_step, we run this loop ever iter
            # - if not every_step, we run this loop every nth iter
            if every_step or (t1 % n_step) == 0:
                meanp, covp = mean1s[t1], cov1s[t1]  # already had to generate h=1 above
                for h in range(1, n_step + 1):
                    if tu + h >= out_timesteps:
                        break
                    if h > 1:
                        meanp, covp = self.ss_step.predict(
                            meanp,
                            covp,
                            {k: v[tu + h] for k, v in predict_kwargs.items()}
                        )
                    if tu + h not in meanps:
                        # idx[tu + h] = tu
                        meanps[tu + h] = meanp
                        covps[tu + h] = covp

        preds = (
            torch.stack([meanps[t] for t in range(out_timesteps)], 1),
            torch.stack([covps[t] for t in range(out_timesteps)], 1)
        )
        updates = torch.stack(meanus, 1), torch.stack(covus, 1)
        R = torch.stack(update_kwargs['R'], 1)
        H = torch.stack(update_kwargs['H'], 1)

        return preds, updates, R, H

    def _build_design_mats(self,
                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                           num_groups: int,
                           out_timesteps: int) -> Tuple[Dict[str, List[Tensor]], Dict[str, List[Tensor]]]:
        """
        Build the design matrices. Implemented by subclasses (partially implemented by
        ``_build_transition_and_measure_mats()`` and ``_build_measure_var_mats()``, but torchscript
        doesn't currently support ``super()``).

        :param static_kwargs: Keys are ids of ``design_modules()``, values are dictionaries of kwargs that only need
         to be passed once.
        :param time_varying_kwargs: Keys are ids of ``design_modules()``, values are dictionaries of kwargs, whose
         values are lists (passed for each timestep).
        :param num_groups: Number of groups.
        :param out_timesteps: Number of timesteps.
        :return: Two dictionaries: ``predict_kwargs`` (passed to ``ss_step.predict()``) and
         ``update_kwargs`` (passed to ``ss_step.update()``).
        """
        raise NotImplementedError

    def _build_transition_and_measure_mats(self,
                                           kwargs_per_process: Dict[str, Dict[str, Tensor]],
                                           num_groups: int,
                                           out_timesteps: int) -> Tuple[List[Tensor], List[Tensor]]:
        ms = self._get_measure_scaling()

        Fs = torch.zeros(
            (num_groups, out_timesteps, self.state_rank, self.state_rank),
            dtype=ms.dtype,
            device=ms.device
        )
        Hs = torch.zeros(
            (num_groups, out_timesteps, len(self.measures), self.state_rank),
            dtype=ms.dtype,
            device=ms.device
        )

        for pid, process in self.processes.items():
            _process_slice = slice(*self.process_to_slice[pid])
            p_kwargs = kwargs_per_process.get(pid, {})
            pH, pF = process(inputs=p_kwargs, num_groups=num_groups, num_times=out_timesteps)
            Hs[:, :, self.measure_to_idx[process.measure], _process_slice] = pH
            Fs[:, :, _process_slice, _process_slice] = pF

        Fs = Fs.unbind(1)
        Hs = Hs.unbind(1)

        return Fs, Hs

    @torch.jit.ignore()
    def _parse_design_kwargs(self, input: Optional[Tensor], out_timesteps: int, **kwargs) -> Dict[str, dict]:
        kwargs_per_process = defaultdict(dict)
        unused = set(kwargs)
        kwargs.update(input=input, current_timestep=torch.tensor(list(range(out_timesteps))).view(1, -1, 1))
        for submodule_nm, submodule in self.design_modules():
            for found_key, key_name, value in get_owned_kwargs(submodule, kwargs):
                unused.discard(found_key)
                kwargs_per_process[submodule_nm][key_name] = value
        if unused:
            warn(f"There are unused keyword arguments:\n{unused}")
        return dict(kwargs_per_process)

    def _get_measure_scaling(self) -> Tensor:
        mcov = self.measure_covariance({}, num_groups=1, num_times=1, _ignore_input=True)[0, 0]
        if self._scale_by_measure_var:
            measure_var = mcov.diagonal(dim1=-2, dim2=-1)
            multi = torch.zeros(mcov.shape[0:-2] + (self.state_rank,), dtype=mcov.dtype, device=mcov.device)
            for pid, process in self.processes.items():
                pidx = self.process_to_slice[pid]
                multi[..., slice(*pidx)] = measure_var[..., self.measure_to_idx[process.measure]].sqrt().unsqueeze(-1)
            assert (multi > 0).all()
        else:
            multi = torch.ones((self.state_rank,), dtype=mcov.dtype, device=mcov.device)
        return multi

    def __repr__(self) -> str:
        return f'{type(self).__name__}' \
               f'(processes={repr(list(self.processes.values()))}, measures={repr(list(self.measures))})'

    @torch.no_grad()
    @torch.jit.ignore()
    def simulate(self,
                 out_timesteps: int,
                 initial_state: Tuple[Optional[Tensor], Optional[Tensor]] = (None, None),
                 start_offsets: Optional[Sequence] = None,
                 num_sims: Optional[int] = None,
                 progress: bool = False,
                 **kwargs):
        """
        Generate simulated state-trajectories from your model.

        :param out_timesteps: The number of timesteps to generate in the output.
        :param initial_state: The initial state of the system: a tuple of `mean`, `cov`.
        :param start_offsets: If your model includes seasonal processes, then these needs to know the start-time for
         each group in ``input``. If you passed ``dt_unit`` when constructing those processes, then you should pass an
         array datetimes here. Otherwise you can pass an array of integers (or leave `None` if there are no seasonal
         processes).
        :param num_sims: The number of state-trajectories to simulate.
        :param progress: Should a progress-bar be displayed? Requires `tqdm`.
        :param kwargs: Further arguments passed to the `processes`.
        :return: A :class:`.Simulations` object with a :func:`Simulations.sample()` method.
        """

        mean, cov = self._prepare_initial_state(initial_state, start_offsets=start_offsets)

        times = range(out_timesteps)
        if progress:
            if progress is True:
                try:
                    from tqdm.auto import tqdm
                    progress = tqdm
                except ImportError:
                    warn("`progress=True` requires package `tqdm`.")
                    progress = lambda x: x
            times = progress(times)

        predict_kwargs, update_kwargs = self._build_design_mats(
            num_groups=num_sims,
            out_timesteps=out_timesteps,
            kwargs_per_process=self._parse_design_kwargs(input=None, out_timesteps=out_timesteps, **kwargs)
        )

        dist_cls = self.ss_step.get_distribution()

        means: List[Tensor] = []
        for t in times:
            mean = dist_cls(mean, cov).rsample()
            mean, cov = self.ss_step.predict(
                mean=mean, cov=.0001 * torch.eye(mean.shape[-1]), kwargs={k: v[t] for k, v in predict_kwargs.items()}
            )
            means.append(mean)

        smeans = torch.stack(means, 1)
        num_groups, num_times, sdim = smeans.shape
        scovs = torch.zeros((num_groups, num_times, sdim, sdim), dtype=smeans.dtype, device=smeans.device)

        return self._generate_predictions(
            preds=(smeans, scovs),
            R=torch.stack(update_kwargs['R'], 1),
            H=torch.stack(update_kwargs['H'], 1)
        )
