from typing import Tuple, Union, Optional, Dict, Iterator, Sequence
from warnings import warn

import torch
from torch import nn, Tensor

import numpy as np

from backports.cached_property import cached_property

from torchcast.internals.utils import get_nan_groups, is_near_zero

from torchcast.utils.data import TimeSeriesDataset


class Predictions(nn.Module):
    """
    The output of the :class:`.StateSpaceModel` forward pass, containing the underlying state means and covariances, as
    well as the predicted observations and covariances.
    """

    def __init__(self,
                 state_means: Sequence[Tensor],
                 state_covs: Sequence[Tensor],
                 R: Sequence[Tensor],
                 H: Sequence[Tensor],
                 model: Union['StateSpaceModel', dict],
                 update_means: Optional[Sequence[Tensor]] = None,
                 update_covs: Optional[Sequence[Tensor]] = None):
        super().__init__()

        # predictions state:
        self._state_means = state_means
        self._state_covs = state_covs

        # updates state:
        self._update_means = update_means
        self._update_covs = update_covs

        # design mats:
        self._H = H
        self._R = R

        # some model attributes are needed for `log_prob` method and for names for plotting
        if not isinstance(model, dict):
            all_state_elements = []
            for pid in model.processes:
                process = model.processes[pid]
                for state_element in process.state_elements:
                    all_state_elements.append((pid, state_element))
            model = {
                'distribution_cls': model.ss_step.get_distribution(),
                'measures': model.measures,
                'all_state_elements': all_state_elements
            }
        self.distribution_cls = model['distribution_cls']
        self.measures = model['measures']
        self.all_state_elements = model['all_state_elements']

        # for lazily populated properties:
        self._means = self._covs = None

        # useful to have:
        self.num_groups, self.num_timesteps, self.state_size = self.state_means.shape

    @cached_property
    def R(self) -> torch.Tensor:
        if not isinstance(self._R, torch.Tensor):
            self._R = torch.stack(self._R, 1)
        return self._R

    @cached_property
    def H(self) -> torch.Tensor:
        if not isinstance(self._H, torch.Tensor):
            self._H = torch.stack(self._H, 1)
        return self._H

    @cached_property
    def state_means(self) -> torch.Tensor:
        if not isinstance(self._state_means, torch.Tensor):
            self._state_means = torch.stack(self._state_means, 1)
        if torch.isnan(self._state_means).any():
            raise ValueError("`nans` in `state_means`")
        return self._state_means

    @cached_property
    def state_covs(self) -> torch.Tensor:
        if not isinstance(self._state_covs, torch.Tensor):
            self._state_covs = torch.stack(self._state_covs, 1)
        if torch.isnan(self._state_covs).any():
            raise ValueError("`nans` in `state_covs`")
        return self._state_covs

    @cached_property
    def update_means(self) -> Optional[torch.Tensor]:
        if self._update_means is None:
            return None
        if not isinstance(self._update_means, torch.Tensor):
            self._update_means = torch.stack(self._update_means, 1)
        if torch.isnan(self._update_means).any():
            raise ValueError("`nans` in `state_means`")
        return self._update_means

    @cached_property
    def update_covs(self) -> torch.Tensor:
        if self._update_covs is None:
            return None
        if not isinstance(self._update_covs, torch.Tensor):
            self._update_covs = torch.stack(self._update_covs, 1)
        if torch.isnan(self._update_covs).any():
            raise ValueError("`nans` in `update_covs`")
        return self._update_covs

    def get_state_at_times(self,
                           times: Union[np.ndarray, np.datetime64],
                           start_times: Optional[np.ndarray] = None,
                           dt_unit: Optional[str] = None,
                           type_: str = 'update') -> Tuple[Tensor, Tensor]:
        """
        For each group, get the state (tuple of (mean, cov)) for a timepoint. This is often useful since predictions
        are right-aligned and padded, so that the final prediction for each group is arbitrarily padded and does not
        correspond to a timepoint of interest -- e.g. for forecasting (i.e., calling
        ``StateSpaceModel.forward(initial_state=get_state_at_times(...))``).

        :param times: Either (a) indices corresponding to each group (e.g. ``times[0]`` corresponds to the timestep to
         take for the 0th group, ``times[1]`` the timestep to take for the 1th group, etc.) or (b) if ``start_times``
         is passed, an array of datetimes. Will also support a single datetime.
        :param start_times: If ``times`` is an array of datetimes, must also pass ``start_datetimes``, i.e. the
         datetimes at which each group started.
        :param dt_unit: If ``times`` is an array of datetimes, must also pass ``dt_unit``, i.e. a
         :class:`numpy.timedelta64` that indicates how much time passes at each timestep. (times-start_times)/dt_unit
         should be an array of integers.
        :param type_: What type of state? Since this method is typically used for getting an `initial_state` for
         another call to :func:`StateSpaceModel.forward()`, this should generally be 'update' (the default); other
         option is 'prediction'.
        :return: A tuple of state-means and state-covs, appropriate for forecasting by passing as `initial_state`
         for :func:`StateSpaceModel.forward()`.
        """
        sliced = self._subset_to_times(times=times, start_times=start_times, dt_unit=dt_unit)
        if type_.startswith('pred'):
            return sliced.state_means.squeeze(1), sliced.state_covs.squeeze(1)
        elif type_.startswith('update'):
            if self.update_means is None:
                raise RuntimeError(
                    "Cannot get with ``type_='update'`` because update mean/cov was not passed when creating this "
                    "``Predictions`` object. This usually means you have to include ``include_updates=True`` when "
                    "calling ``StateSpaceModel``."
                )
            return sliced.update_means.squeeze(1), sliced.update_covs.squeeze(1)
        else:
            raise ValueError("Unrecognized `type_`, expected 'prediction' or 'update'.")

    @classmethod
    def observe(cls, state_means: Tensor, state_covs: Tensor, R: Tensor, H: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Convert latent states into observed predictions (and their uncertainty).

        :param state_means: The latent state means
        :param state_covs: The latent state covs.
        :param R: The measure-covariance matrices.
        :param H: The measurement matrix.
        :return: A tuple of `means`, `covs`.
        """
        means = H.matmul(state_means.unsqueeze(-1)).squeeze(-1)
        pargs = list(range(len(H.shape)))
        pargs[-2:] = reversed(pargs[-2:])
        Ht = H.permute(*pargs)
        assert R.shape[-1] == R.shape[-2], f"R is not symmetrical (shape is {R.shape})"
        covs = H.matmul(state_covs).matmul(Ht) + R
        return means, covs

    @property
    def means(self) -> Tensor:
        if self._means is None:
            # TODO: in ExpSmooth, _state_covs, _R, and _H will often not be time-varying. if we could generate them s.t.
            #  we could perform a fast check of this (self._R[0] is self._R[1]) then could speed up slowest step:
            #  `H.matmul(state_covs).matmul(Ht) + R`
            self._means, self._covs = self.observe(self.state_means, self.state_covs, self.R, self.H)
        return self._means

    @property
    def covs(self) -> Tensor:
        if self._covs is None:
            self._means, self._covs = self.observe(self.state_means, self.state_covs, self.R, self.H)
        return self._covs

    def sample(self) -> Tensor:
        with torch.no_grad():
            dist = self.distribution_cls(self.means, self.covs)
            return dist.rsample()

    def log_prob(self, obs: Tensor) -> Tensor:
        """
        Compute the log-probability of data (e.g. data that was originally fed into the KalmanFilter).

        :param obs: A Tensor that could be used in the KalmanFilter.forward pass.
        :return: A tensor with one element for each group X timestep indicating the log-probability.
        """
        assert len(obs.shape) == 3
        assert obs.shape[-1] == self.means.shape[-1]
        n_measure_dim = obs.shape[-1]
        n_state_dim = self.state_means.shape[-1]

        obs_flat = obs.reshape(-1, n_measure_dim)
        state_means_flat = self.state_means.view(-1, n_state_dim)
        state_covs_flat = self.state_covs.view(-1, n_state_dim, n_state_dim)
        H_flat = self.H.view(-1, n_measure_dim, n_state_dim)
        R_flat = self.R.view(-1, n_measure_dim, n_measure_dim)

        lp_flat = torch.zeros(obs_flat.shape[0], dtype=self.state_means.dtype, device=self.state_means.device)
        for gt_idx, valid_idx in get_nan_groups(torch.isnan(obs_flat)):
            if valid_idx is None:
                gt_obs = obs_flat[gt_idx]
                gt_means_flat = self.means.view(-1, n_measure_dim)[gt_idx]
                gt_covs_flat = self.covs.view(-1, n_measure_dim, n_measure_dim)[gt_idx]
            else:
                mask1d = torch.meshgrid(gt_idx, valid_idx, indexing='ij')
                mask2d = torch.meshgrid(gt_idx, valid_idx, valid_idx, indexing='ij')
                gt_means_flat, gt_covs_flat = self.observe(
                    state_means=state_means_flat[gt_idx],
                    state_covs=state_covs_flat[gt_idx],
                    R=R_flat[mask2d],
                    H=H_flat[mask1d]
                )
                gt_obs = obs_flat[mask1d]
            lp_flat[gt_idx] = self._log_prob(gt_obs, gt_means_flat, gt_covs_flat)

        return lp_flat.view(obs.shape[0:2])

    def _log_prob(self, obs: Tensor, means: Tensor, covs: Tensor) -> Tensor:
        return self.distribution_cls(means, covs, validate_args=False).log_prob(obs)

    def to_dataframe(self,
                     dataset: Union[TimeSeriesDataset, dict],
                     type: str = 'predictions',
                     group_colname: str = 'group',
                     time_colname: str = 'time',
                     multi: Optional[float] = 1.96) -> 'DataFrame':
        """
        :param dataset: Either a :class:`.TimeSeriesDataset`, or a dictionary with 'start_times', 'group_names', &
         'dt_unit'
        :param type: Either 'predictions' or 'components'.
        :param group_colname: Column-name for 'group'
        :param time_colname: Column-name for 'time'
        :param multi: Multiplier on std-dev for lower/upper CIs. Default 1.96.
        :return: A pandas DataFrame with group, 'time', 'measure', 'mean', 'lower', 'upper'. For ``type='components'``
         additionally includes: 'process' and 'state_element'.
        """

        from pandas import concat

        if isinstance(dataset, TimeSeriesDataset):
            batch_info = {
                'start_times': dataset.start_times,
                'group_names': dataset.group_names,
                'named_tensors': {},
                'dt_unit': dataset.dt_unit
            }
            for measure_group, tensor in zip(dataset.measures, dataset.tensors):
                for i, measure in enumerate(measure_group):
                    if measure in self.measures:
                        batch_info['named_tensors'][measure] = tensor[..., [i]]
            missing = set(self.measures) - set(dataset.all_measures)
            if missing:
                raise ValueError(
                    f"Some measures in the design aren't in the dataset.\n"
                    f"Design: {missing}\nDataset: {dataset.all_measures}"
                )
        elif isinstance(dataset, dict):
            batch_info = dataset.copy()
            if isinstance(batch_info['dt_unit'], str):
                batch_info['dt_unit'] = np.timedelta64(1, batch_info['dt_unit'])
        else:
            raise TypeError(
                "Expected `batch` to be a TimeSeriesDataset, or a dictionary with 'start_times' and 'group_names'."
            )

        def _tensor_to_df(tens, measures):
            offsets = np.arange(0, tens.shape[1]) * (batch_info['dt_unit'] if batch_info['dt_unit'] else 1)
            times = batch_info['start_times'][:, None] + offsets

            return TimeSeriesDataset.tensor_to_dataframe(
                tensor=tens,
                times=times,
                group_names=batch_info['group_names'],
                group_colname=group_colname,
                time_colname=time_colname,
                measures=measures
            )

        assert group_colname not in {'mean', 'lower', 'upper', 'std'}
        assert time_colname not in {'mean', 'lower', 'upper', 'std'}

        out = []
        if type == 'predictions':

            stds = torch.diagonal(self.covs, dim1=-1, dim2=-2).sqrt()
            for i, measure in enumerate(self.measures):
                # predicted:
                df = _tensor_to_df(torch.stack([self.means[..., i], stds[..., i]], 2), measures=['mean', 'std'])
                if multi is not None:
                    df['lower'] = df['mean'] - multi * df['std']
                    df['upper'] = df['mean'] + multi * df.pop('std')

                # actual:
                orig_tensor = batch_info.get('named_tensors', {}).get(measure, None)
                if orig_tensor is not None and (orig_tensor == orig_tensor).any():
                    df_actual = _tensor_to_df(orig_tensor, measures=['actual'])
                    df = df.merge(df_actual, on=[group_colname, time_colname], how='left')

                out.append(df.assign(measure=measure))

        elif type == 'components':
            # components:
            for (measure, process, state_element), (m, std) in self._components().items():
                df = _tensor_to_df(torch.stack([m, std], 2), measures=['mean', 'std'])
                if multi is not None:
                    df['lower'] = df['mean'] - multi * df['std']
                    df['upper'] = df['mean'] + multi * df.pop('std')
                df['process'], df['state_element'], df['measure'] = process, state_element, measure
                out.append(df)

            # residuals:
            named_tensors = batch_info.get('named_tensors', {})
            for i, measure in enumerate(self.measures):
                orig_tensor = named_tensors.get(measure)
                predictions = self.means[..., [i]]
                if orig_tensor.shape[1] < predictions.shape[1]:
                    orig_aligned = predictions.data.clone()
                    orig_aligned[:] = float('nan')
                    orig_aligned[:, 0:orig_tensor.shape[1], :] = orig_tensor
                else:
                    orig_aligned = orig_tensor[:, 0:predictions.shape[1], :]

                df = _tensor_to_df(predictions - orig_aligned, ['mean'])
                df['process'], df['state_element'], df['measure'] = 'residuals', 'residuals', measure
                out.append(df)

        else:
            raise ValueError("Expected `type` to be 'predictions' or 'components'.")

        return concat(out, sort=True)

    @torch.no_grad()
    def _components(self) -> Dict[Tuple[str, str, str], Tuple[Tensor, Tensor]]:
        out = {}
        for midx, measure in enumerate(self.measures):
            H = self.H[..., midx, :]
            means = H * self.state_means
            stds = H * torch.diagonal(self.state_covs, dim1=-2, dim2=-1).sqrt()

            for se_idx, (process, state_element) in enumerate(self.all_state_elements):
                if not is_near_zero(means[:, :, se_idx]).all():
                    out[(measure, process, state_element)] = (means[:, :, se_idx], stds[:, :, se_idx])

        return out

    @staticmethod
    def plot(df: 'DataFrame',
             group_colname: str = None,
             time_colname: str = None,
             max_num_groups: int = 1,
             split_dt: Optional[np.datetime64] = None,
             **kwargs) -> 'DataFrame':
        """
        :param df: The output of :func:`Predictions.to_dataframe()`.
        :param group_colname: The name of the group-column.
        :param time_colname: The name of the time-column.
        :param max_num_groups: Max. number of groups to plot; if the number of groups in the dataframe is greater than
         this, a random subset will be taken.
        :param split_dt: If supplied, will draw a vertical line at this date (useful for showing pre/post validation).
        :param kwargs: Further keyword arguments to pass to ``plotnine.theme`` (e.g. ``figure_size=(x,y)``)
        :return: A plot of the predicted and actual values.
        """

        from plotnine import (
            ggplot, aes, geom_line, geom_ribbon, facet_grid, facet_wrap, theme_bw, theme, ylab, geom_vline
        )

        is_components = 'process' in df.columns
        if is_components and 'state_element' not in df.columns:
            df = df.assign(state_element='all')

        if group_colname is None:
            group_colname = 'group'
            if group_colname not in df.columns:
                raise TypeError("Please specify group_colname")
        if time_colname is None:
            time_colname = 'time'
            if 'time' not in df.columns:
                raise TypeError("Please specify time_colname")

        df = df.copy()
        if 'upper' not in df.columns and 'std' in df.columns:
            df['upper'] = df['mean'] + 1.96 * df['std']
            df['lower'] = df['mean'] - 1.96 * df['std']
        if df[group_colname].nunique() > max_num_groups:
            subset_groups = df[group_colname].drop_duplicates().sample(max_num_groups).tolist()
            if len(subset_groups) < df[group_colname].nunique():
                print("Subsetting to groups: {}".format(subset_groups))
            df = df.loc[df[group_colname].isin(subset_groups), :]
        num_groups = df[group_colname].nunique()

        aes_kwargs = {'x': time_colname}
        if is_components:
            aes_kwargs['group'] = 'state_element'

        plot = (
                ggplot(df, aes(**aes_kwargs)) +
                geom_line(aes(y='mean'), color='#4C6FE7', size=1.5, alpha=.75) +
                geom_ribbon(aes(ymin='lower', ymax='upper'), color=None, alpha=.25) +
                ylab("")
        )

        assert 'measure' in df.columns
        if is_components:
            num_processes = df['process'].nunique()
            if num_groups > 1 and num_processes > 1:
                raise ValueError("Cannot plot components for > 1 group and > 1 processes.")
            elif num_groups == 1:
                plot = plot + facet_wrap(f"~ measure + process", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    from plotnine.facets.facet_wrap import n2mfrow
                    nrow, _ = n2mfrow(len(df[['process', 'measure']].drop_duplicates().index))
                    kwargs['figure_size'] = (12, nrow * 2.5)
            else:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
                if 'figure_size' not in kwargs:
                    kwargs['figure_size'] = (12, num_groups * 2.5)

            if (df.groupby('measure')['process'].nunique() <= 1).all():
                plot = plot + geom_line(aes(y='mean', color='state_element'), size=1.5)

        else:
            if 'actual' in df.columns:
                plot = plot + geom_line(aes(y='actual'))
            if num_groups > 1:
                plot = plot + facet_grid(f"{group_colname} ~ measure", scales='free_y', labeller='label_both')
            else:
                plot = plot + facet_wrap("~measure", scales='free_y', labeller='label_both')

            if 'figure_size' not in kwargs:
                kwargs['figure_size'] = (12, 5)

        if split_dt:
            plot = plot + geom_vline(xintercept=np.datetime64(split_dt), linetype='dashed')

        return plot + theme_bw() + theme(**kwargs)

    def _subset_to_times(self,
                         times: Union[np.ndarray, np.datetime64],
                         start_times: Optional[np.ndarray] = None,
                         dt_unit: Optional[str] = None) -> 'Predictions':
        """
        Return a `Predictions` object with a single timepoint for each group.
        """
        if not isinstance(times, (list, tuple, np.ndarray)):
            times = np.asanyarray([times] * self.num_groups)

        if start_times is not None:
            if isinstance(dt_unit, str):
                dt_unit = np.datetime64(1, dt_unit)
            times = times - start_times
            if dt_unit is not None:
                times = times // dt_unit  # todo: validate int?

        assert len(times.shape) == 1
        assert times.shape[0] == self.num_groups
        idx = (torch.arange(self.num_groups), torch.as_tensor(times, dtype=torch.int64))
        return self._subset(idx, collapsed_dim=1)

    def __iter__(self) -> Iterator[Tensor]:
        # for mean, cov = tuple(predictions)
        yield self.means
        yield self.covs

    def __array__(self) -> np.ndarray:
        # for numpy.asarray
        return self.means.detach().numpy()

    def __getitem__(self, item: Tuple) -> 'Predictions':
        return self._subset(item)

    def _subset(self, idx: Tuple, collapsed_dim: Optional[int] = None) -> 'Predictions':
        """
        Helper for __getitem__ and get_timeslice
        """
        if collapsed_dim is not None:
            assert collapsed_dim < 2
        kwargs = {
            'state_means': self.state_means[idx],
            'state_covs': self.state_covs[idx],
            'H': self.H[idx],
            'R': self.R[idx]
        }
        if self.update_means is not None:
            kwargs.update({'update_means': self.update_means[idx], 'update_covs': self.update_covs[idx]})
        cls = type(self)
        for k in list(kwargs):
            expected_shape = getattr(self, k).shape
            if collapsed_dim is not None:
                kwargs[k] = kwargs[k].unsqueeze(collapsed_dim)
            v = kwargs[k]
            if len(v.shape) != len(expected_shape):
                raise TypeError(f"Expected {k} to have shape {expected_shape} but got {v.shape}.")
            if v.shape[-1] != expected_shape[-1]:
                raise TypeError(f"Cannot index into non-batch dims of {cls.__name__}")
            if k == 'H' and v.shape[-2] != self.H.shape[-2]:
                raise TypeError(f"Cannot index into non-batch dims of {cls.__name__}")
        return cls(**kwargs, model=self._model_attributes)

    @property
    def _model_attributes(self) -> dict:
        """
        Has the attributes of a KalmanFilter that are needed in __init__
        """
        return dict(
            measures=self.measures,
            distribution_cls=self.distribution_cls,
            all_state_elements=self.all_state_elements
        )
