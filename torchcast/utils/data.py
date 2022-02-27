import datetime
import itertools

from typing import Sequence, Any, Union, Optional, Tuple, Callable
from warnings import warn

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset, Dataset

from torchcast.internals.utils import ragged_cat, true1d_idx


class TimeSeriesDataset(TensorDataset):
    """
    :class:`.TimeSeriesDataset` includes additional information about each of the Tensors' dimensions: the name for
    each group in the first dimension, the start (date)time (and optionally datetime-unit) for the second dimension,
    and the name of the measures for the third dimension.

    Note that unlike :class:`torch.utils.data.TensorDataset`, indexing a :class:`.TimeSeriesDataset` returns another
    :class:`.TimeSeriesDataset`, not a tuple of tensors. So when using :class:`.TimeSeriesDataset`, use
    :class:`.TimeSeriesDataLoader` (equivalent to ``DataLoader(collate_fn=TimeSeriesDataset.collate)``).
    """
    _repr_attrs = ('sizes', 'measures')

    def __init__(self,
                 *tensors: Tensor,
                 group_names: Sequence[Any],
                 start_times: Union[np.ndarray, Sequence],
                 measures: Sequence[Sequence[str]],
                 dt_unit: Optional[str]):

        if not isinstance(group_names, np.ndarray):
            group_names = np.array(group_names)

        assert len(group_names) == len(set(group_names))
        assert len(group_names) == len(start_times)
        assert len(tensors) == len(measures)

        for i, (tensor, tensor_measures) in enumerate(zip(tensors, measures)):
            if len(tensor.shape) < 3:
                raise ValueError(f"Tensor {i} has < 3 dimensions")
            if tensor.shape[0] != len(group_names):
                raise ValueError(f"Tensor {i}'s first dimension has length != {len(group_names)}.")
            if tensor.shape[2] != len(tensor_measures):
                raise ValueError(f"Tensor {i}'s 3rd dimension has length != len({tensor_measures}).")

        self.measures = tuple(tuple(m) for m in measures)
        self.all_measures = tuple(itertools.chain.from_iterable(self.measures))
        self.group_names = group_names
        self.dt_unit = None
        if dt_unit:
            if not isinstance(dt_unit, np.timedelta64):
                dt_unit = np.timedelta64(1, dt_unit)
            self.dt_unit = dt_unit
        start_times = np.asanyarray(start_times)
        if self.dt_unit:
            assert len(start_times.shape) == 1
            if isinstance(start_times[0], (np.datetime64, datetime.datetime)):
                start_times = np.array(start_times, dtype='datetime64')
            else:
                raise ValueError("`dt_unit` is not None but `start_times` is not an array of datetimes")
        else:
            if not isinstance(start_times[0], int) and not float(start_times[0]).is_integer():
                raise ValueError(
                    f"`dt_unit` is None but `start_times` does not appear to be integers "
                    f"(e.g. start_times[0] is {start_times[0]})."
                )
        self.start_times = start_times
        super().__init__(*tensors)

    def to(self, *args, **kwargs) -> 'TimeSeriesDataset':
        new_tensors = [x.to(*args, **kwargs) for x in self.tensors]
        return self.with_new_tensors(*new_tensors)

    def __repr__(self) -> str:
        kwargs = []
        for k in self._repr_attrs:
            v = getattr(self, k)
            if isinstance(v, Tensor):
                v = v.size()
            kwargs.append("{}={!r}".format(k, v))
        return "{}({})".format(type(self).__name__, ", ".join(kwargs))

    @property
    def sizes(self) -> Sequence:
        return [t.size() for t in self.tensors]

    # Subsetting ------------------------:
    @torch.no_grad()
    def train_val_split(self,
                        train_frac: float = None,
                        dt: Union[np.datetime64, dict] = None,
                        quiet: bool = False) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """
        :param train_frac: The proportion of the data to keep for training. This is calculated on a per-group basis, by
         taking the last observation for each group (i.e., the last observation that a non-nan value on any measure). If
         neither `train_frac` nor `dt` are passed, ``train_frac=.75`` is used.
        :param dt: A datetime to use in dividing train/validation (first datetime for validation), or a dictionary of
         group-names : date-times.
        :return: Two ``TimeSeriesDatasets``, one with data before the split, the other with >= the split.
        """

        # get split times:
        if dt is None:
            if train_frac is None:
                train_frac = .75
            assert 0 < train_frac < 1
            # for each group, find the last non-nan, take `frac` of that to find the train/val split point:
            split_idx = np.array([int(idx * train_frac) for idx in self._last_measured_idx()], dtype='int')
            _times = self.times(0)
            split_times = np.array([_times[i, t] for i, t in enumerate(split_idx)])
        else:
            if train_frac is not None:
                raise TypeError("Can pass only one of `train_frac`, `dt`.")
            if isinstance(dt, dict):
                split_times = np.array([dt[group_name] for group_name in self.group_names],
                                       dtype='datetime64[ns]' if self.dt_unit else 'int')
            else:
                if self.dt_unit:
                    if isinstance(dt, datetime.datetime):  # base
                        dt = np.datetime64(dt)
                    if hasattr(dt, 'to_datetime64'):  # pandas
                        dt = dt.to_datetime64()
                    if not isinstance(dt, np.datetime64):
                        raise ValueError(f"`dt` is not datetimelike, but dataset has non-null dt_unit `{self.dt_unit}`")
                split_times = np.full(shape=len(self.group_names), fill_value=dt)

        # val:
        val_dataset = self.with_new_start_times(split_times, quiet=quiet)

        # train:
        train_tensors = []
        for i, tens in enumerate(self.tensors):
            train = tens.clone()
            train[np.where(self.times(i) >= split_times[:, None])] = float('nan')
            if i == 0:
                not_all_nan = (~torch.isnan(train)).sum((0, 2))
                last_good_idx = true1d_idx(not_all_nan).max()
            train = train[:, :(last_good_idx + 1), :]
            train_tensors.append(train)
        # TODO: replace padding nans for all but first tensor?
        # TODO: reduce width of 0> tensors based on width of 0 tensor?
        train_dataset = self.with_new_tensors(*train_tensors)

        return train_dataset, val_dataset

    def with_new_start_times(self,
                             start_times: Union[np.ndarray, Sequence],
                             quiet: bool = False) -> 'TimeSeriesDataset':
        """
        Subset a :class:`.TimeSeriesDataset` so that some/all of the groups have later start times.

        :param start_times: An array/list of new datetimes.
        :param quiet: If True, will not emit a warning for groups having only `nan` after the start-time.
        :return: A new :class:`.TimeSeriesDataset`.
        """
        new_tensors = []
        for i, tens in enumerate(self.tensors):
            times = self.times(i)
            new_tens = []
            for g, (new_time, old_times) in enumerate(zip(start_times, times)):
                if (old_times <= new_time).all():
                    warn(f"{new_time} is later than all the times for group {self.group_names[g]}")
                    new_tens.append(tens[[g], 0:0])
                    continue
                elif (old_times > new_time).all():
                    warn(f"{new_time} is earlier than all the times for group {self.group_names[g]}")
                    new_tens.append(tens[[g], 0:0])
                    continue
                # drop if before new_time:
                g_tens = tens[g, true1d_idx(old_times >= new_time)]
                # drop if after last nan:
                all_nan, _ = torch.min(torch.isnan(g_tens), 1)
                if all_nan.all():
                    if not quiet:
                        warn(f"Group '{self.group_names[g]}' (tensor {i}) has only `nans` after {new_time}")
                    end_idx = 0
                else:
                    end_idx = true1d_idx(~all_nan).max() + 1
                new_tens.append(g_tens[:end_idx].unsqueeze(0))
            new_tens = ragged_cat(new_tens, ragged_dim=1, cat_dim=0)
            new_tensors.append(new_tens)
        return type(self)(
            *new_tensors,
            group_names=self.group_names,
            start_times=start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    def get_groups(self, groups: Sequence[Any]) -> 'TimeSeriesDataset':
        """
        Get the subset of the batch corresponding to groups. Note that the ordering in the output will match the
        original ordering (not that of `group`), and that duplicates will be dropped.
        """
        return self[np.isin(self.group_names, groups)]

    def split_measures(self, *measure_groups, which: Optional[int] = None) -> 'TimeSeriesDataset':
        """
        Take a dataset with one tensor, split it into a dataset with multiple tensors.

        :param measure_groups: Each argument should be be a list of measure-names, or an indexer (i.e. list of ints or
         a slice).
        :param which: If there are already multiple measure groups, the split will occur within one of them; must
         specify which.
        :return: A :class:`.TimeSeriesDataset`, now with multiple tensors for the measure-groups.
        """

        if which is None:
            if len(self.measures) > 1:
                raise RuntimeError(f"Must pass `which` if there's more than one groups:\n{self.measures}")
            which = 0

        self_tensor = self.tensors[which]
        self_measures = self.measures[which]

        idxs = []
        for measure_group in measure_groups:
            if isinstance(measure_group, slice) or isinstance(measure_group[0], int):
                idxs.append(measure_group)
            else:
                idxs.append([self_measures.index(m) for m in measure_group])

        self_measures = np.array(self_measures)
        return type(self)(
            *(self_tensor[:, :, idx].clone() for idx in idxs),
            start_times=self.start_times,
            group_names=self.group_names,
            measures=[tuple(self_measures[idx]) for idx in idxs],
            dt_unit=self.dt_unit
        )

    def __getitem__(self, item: Union[int, Sequence, slice]) -> 'TimeSeriesDataset':
        if isinstance(item, int):
            item = [item]
        return type(self)(
            *super(TimeSeriesDataset, self).__getitem__(item),
            group_names=self.group_names[item],
            start_times=self.start_times[item],
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    # Creation/Transformation ------------------------:
    @classmethod
    def make_collate_fn(cls, pad_X: Optional[float] = 0.) -> Callable:
        def collate_fn(batch: Sequence['TimeSeriesDataset']) -> 'TimeSeriesDataset':
            to_concat = {
                'tensors': [batch[0].tensors],
                'group_names': [batch[0].group_names],
                'start_times': [batch[0].start_times]
            }
            fixed = {'dt_unit': batch[0].dt_unit, 'measures': batch[0].measures}
            for i, ts_dataset in enumerate(batch[1:], 1):
                for attr, appendlist in to_concat.items():
                    to_concat[attr].append(getattr(ts_dataset, attr))
                for attr, required_val in fixed.items():
                    new_val = getattr(ts_dataset, attr)
                    if new_val != required_val:
                        raise ValueError(
                            f"Element {i} has `{attr}` = {new_val}, but for element 0 it's {required_val}."
                        )

            tensors = []
            for i, t in enumerate(zip(*to_concat['tensors'])):
                tensors.append(ragged_cat(t, ragged_dim=1, padding=None if i == 0 else pad_X))

            return cls(
                *tensors,
                group_names=np.concatenate(to_concat['group_names']),
                start_times=np.concatenate(to_concat['start_times']),
                measures=fixed['measures'],
                dt_unit=fixed['dt_unit']
            )

        return collate_fn

    def to_dataframe(self,
                     group_colname: str = 'group',
                     time_colname: str = 'time'
                     ) -> 'DataFrame':

        return self.tensor_to_dataframe(
            tensor=ragged_cat(self.tensors, ragged_dim=1, cat_dim=2),
            times=self.times(),
            group_names=self.group_names,
            group_colname=group_colname,
            time_colname=time_colname,
            measures=self.all_measures
        )

    @staticmethod
    @torch.no_grad()
    def tensor_to_dataframe(tensor: Tensor,
                            times: np.ndarray,
                            group_names: Sequence,
                            group_colname: str,
                            time_colname: str,
                            measures: Sequence[str]) -> 'DataFrame':
        from pandas import DataFrame, concat

        tensor = tensor.cpu().numpy()
        assert tensor.shape[0] == len(group_names)
        assert tensor.shape[0] == len(times)
        assert tensor.shape[1] <= times.shape[1]
        assert tensor.shape[2] == len(measures)

        dfs = []
        for g, group_name in enumerate(group_names):
            # get values, don't store trailing nans:
            values = tensor[g]
            all_nan_per_row = np.min(np.isnan(values), axis=1)
            if all_nan_per_row.all():
                warn(f"Group {group_name} has only missing values.")
                continue
            end_idx = true1d_idx(~all_nan_per_row).max() + 1
            # convert to dataframe:
            df = DataFrame(data=values[:end_idx, :], columns=measures)
            df[group_colname] = group_name
            df[time_colname] = np.nan
            df[time_colname] = times[g, 0:len(df.index)]
            dfs.append(df)
        if dfs:
            return concat(dfs)
        else:
            return DataFrame(columns=list(measures) + [group_colname, time_colname])

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: Optional[str],
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Sequence[str]] = None,
                       y_colnames: Optional[Sequence[str]] = None,
                       pad_X: Optional[float] = 0.,
                       **kwargs) -> 'TimeSeriesDataset':
        """
        :param dataframe: A pandas ``DataFrame``
        :param group_colname: Name for the group-column name.
        :param time_colname: Name for the time-column name.
        :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
         used -- i.e., how far we advance with every timestep. Can be `None` if the data are in arbitrary (non-datetime)
         units.
        :param measure_colnames: A list of names of columns that include the actual time-series data in the dataframe.
         Optional if `X_colnames` and `y_colnames` are passed.
        :param X_colnames: In many settings we have a set of columns corresponding to predictors and a set of columns
         corresponding to the actual time-series data. The former should be passed as `X_colnames` and the latter as
         `y_colnames`.
        :param y_colnames: See above.
        :param pad_X: When stacking time-serieses of unequal length, we left-align them and so get trailing missings.
         Setting ``pad_X`` allows you to select the padding value for these. Default 0-padding.
        :param kwargs: The `dtype` and/or the `device`.
        :return: A :class:`TimeSeriesDataset`.
        """
        if 'dtype' not in kwargs:
            kwargs['dtype'] = torch.float32

        if X_colnames is not None:
            X_colnames = list(X_colnames)
        if y_colnames is not None:
            y_colnames = list(y_colnames)
        if measure_colnames is not None:
            measure_colnames = list(measure_colnames)

        if measure_colnames is None:
            if y_colnames is None or X_colnames is None:
                raise ValueError("Must pass either `measure_colnames` or `X_colnames` & `y_colnames`")
            if isinstance(y_colnames, str):
                y_colnames = [y_colnames]
                warn(f"`y_colnames` should be a list of strings not a string; interpreted as `{y_colnames}`.")

            measure_colnames = list(y_colnames) + list(X_colnames)
        else:
            if X_colnames is not None or y_colnames is not None:
                raise ValueError("If passing `measure_colnames` do not pass `X_colnames` or `y_colnames`.")
            if isinstance(measure_colnames, str):
                measure_colnames = [measure_colnames]
                warn(f"`measure_colnames` should be list of strings not a string; interpreted as `{measure_colnames}`.")

        if group_colname is None:
            dataframe = dataframe.assign(_group=1)
            group_colname = '_group'
        assert isinstance(group_colname, str)
        assert isinstance(time_colname, str)
        assert len(measure_colnames) == len(set(measure_colnames))

        # sort by time:
        dataframe = dataframe.sort_values(time_colname)

        for measure_colname in measure_colnames:
            if measure_colname not in dataframe.columns:
                raise ValueError(f"'{measure_colname}' not in dataframe.columns:\n{dataframe.columns}'")

        # first pass for info:
        arrays, time_idxs, group_names, start_times = [], [], [], []
        for g, df in dataframe.groupby(group_colname, sort=True):
            # group-names:
            group_names.append(g)

            # times:
            times = df[time_colname].values
            assert len(times) == len(set(times)), f"Group {g} has duplicate times"
            min_time = times[0]
            start_times.append(min_time)
            if dt_unit is None:
                time_idx = (times - min_time).astype('int64')
            else:
                if not isinstance(dt_unit, np.timedelta64):
                    dt_unit = np.timedelta64(1, dt_unit)
                time_idx = (times - min_time) // dt_unit
            time_idxs.append(time_idx)

            # values:
            arrays.append(df.loc[:, measure_colnames].values)

        # second pass organizes into tensor
        time_len = max(time_idx[-1] + 1 for time_idx in time_idxs)
        tens = torch.empty((len(arrays), time_len, len(measure_colnames)), **kwargs)
        tens[:] = np.nan
        for i, (array, time_idx) in enumerate(zip(arrays, time_idxs)):
            tens[i, time_idx, :] = torch.tensor(array, **kwargs)

        dataset = cls(
            tens,
            group_names=group_names,
            start_times=start_times,
            measures=[measure_colnames],
            dt_unit=dt_unit
        )

        if X_colnames is not None:
            dataset = dataset.split_measures(y_colnames, X_colnames)
            y, X = dataset.tensors
            # don't use nan-padding on the y tensor:
            if pad_X is not None:
                for i, time_idx in enumerate(time_idxs):
                    pad_idx = time_idx.max() + 1
                    X[i, pad_idx:, :] = pad_X

        return dataset

    def with_new_tensors(self, *tensors: Tensor) -> 'TimeSeriesDataset':
        """
        Create a new Batch with a different Tensor, but all other attributes the same.
        """
        return type(self)(
            *tensors,
            group_names=self.group_names,
            start_times=self.start_times,
            measures=self.measures,
            dt_unit=self.dt_unit
        )

    # Util/Private ------------------------
    def times(self, which: Optional[int] = None) -> np.ndarray:
        """
        A 2D array of datetimes (or integers if dt_unit is None) for this dataset.

        :param which: If this dataset has multiple tensors of different number of timesteps, which should be used for
         constructing the `times` array? Defaults to the one with the most timesteps.
        :return: A 2D numpy array of datetimes (or integers if dt_unit is None).
        """
        if which is None:
            num_timesteps = max(tensor.shape[1] for tensor in self.tensors)
        else:
            num_timesteps = self.tensors[which].shape[1]
        offsets = np.arange(0, num_timesteps) * (self.dt_unit if self.dt_unit else 1)
        return self.start_times[:, None] + offsets

    def datetimes(self) -> np.ndarray:
        return self.times()

    @property
    def start_datetimes(self) -> np.ndarray:
        return self.start_times

    @property
    def start_offsets(self) -> np.ndarray:
        return self.start_times

    def _last_measured_idx(self) -> np.ndarray:
        """
        :return: The indices of the last measurement in the first tensor, where a measurement is any non-nan value in at
         least on dimension.
        """
        tens, *_ = self.tensors
        any_measured_bool = ~np.isnan(tens.numpy()).all(2)
        last_measured_idx = np.array(
            [np.max(true1d_idx(any_measured_bool[g]), initial=0) for g in range(len(self.group_names))],
            dtype='int'
        )
        return last_measured_idx


class TimeSeriesDataLoader(DataLoader):
    """
    This is a convenience wrapper around
    ``DataLoader(collate_fn=TimeSeriesDataset.make_collate_fn())``. Additionally, it provides a ``from_dataframe()``
    classmethod so that the data-loader can be created directly from a pandas dataframe. This can be more
    memory-efficient than the alternative route of first creating a :class:`.TimeSeriesDataset` from a dataframe, and
    then passing that object to a data-loader.
    """

    def __init__(self,
                 dataset: 'Dataset',
                 batch_size: Optional[int],
                 pad_X: Optional[float] = 0.,
                 **kwargs):
        """
        :param dataset: A TimeSeriesDataset
        :param batch_size: Series per batch to load.
        :param pad_X: When stacking time-serieses of unequal length, we left-align them and so get trailing nans.
         Setting ``pad_X`` allows you to select the padding value for these.
        :param kwargs: Other arguments passed to :class:`torch.utils.data.DataLoader`
        """
        kwargs['collate_fn'] = TimeSeriesDataset.make_collate_fn(pad_X)
        super().__init__(dataset=dataset, batch_size=batch_size, **kwargs)

    @classmethod
    def from_dataframe(cls,
                       dataframe: 'DataFrame',
                       group_colname: str,
                       time_colname: str,
                       dt_unit: Optional[str],
                       measure_colnames: Optional[Sequence[str]] = None,
                       X_colnames: Optional[Sequence[str]] = None,
                       y_colnames: Optional[Sequence[str]] = None,
                       pad_X: Optional[float] = 0.,
                       **kwargs) -> 'TimeSeriesDataLoader':
        """
        :param dataframe: A pandas ``DataFrame``
        :param group_colname: Name for the group-column name.
        :param time_colname: Name for the time-column name.
        :param dt_unit: A numpy.timedelta64 (or string that will be converted to one) that indicates the time-units
         used -- i.e., how far we advance with every timestep. Can be `None` if the data are in arbitrary (non-datetime)
         units.
        :param measure_colnames: A list of names of columns that include the actual time-series data in the dataframe.
         Optional if `X_colnames` and `y_colnames` are passed.
        :param X_colnames: In many settings we have a set of columns corresponding to predictors and a set of columns
         corresponding to the actual time-series data. The former should be passed as `X_colnames` and the latter as
         `y_colnames`.
        :param y_colnames: See above.
        :param pad_X: When stacking time-serieses of unequal length, we left-align them and so get trailing missings.
         Setting ``pad_X`` allows you to select the padding value for these. Default 0-padding.
        :param kwargs: Other arguments to pass to :func:`TimeSeriesDataset.from_dataframe()`.
        :return: An iterable that yields :class:`TimeSeriesDataset`.
        """
        _kwargs = {}
        for k in ('device', 'dtype'):
            if k in kwargs:
                _kwargs[k] = kwargs.pop(k)

        dataset = ConcatDataset(
            datasets=[
                TimeSeriesDataset.from_dataframe(
                    dataframe=df,
                    group_colname=group_colname,
                    time_colname=time_colname,
                    measure_colnames=measure_colnames,
                    X_colnames=X_colnames,
                    y_colnames=y_colnames,
                    dt_unit=dt_unit,
                    **_kwargs
                )
                for g, df in dataframe.groupby(group_colname)
            ]
        )
        return cls(dataset=dataset, pad_X=pad_X, **kwargs)


def complete_times(data: 'DataFrame',
                   group_colname: str,
                   time_colname: Optional[str] = None,
                   dt_unit: Optional[str] = None):
    """
    Given a dataframe time-serieses, convert implicit missings within each time-series to explicit missings.

    :param data: A pandas dataframe.
    :param group_colname: The column name for the groups.
    :param time_colname: The column name for the times. Will attempt to guess based on common labels.
    :param dt_unit: A :class:`numpy.datetime64` or string representing the datetime increments. If not supplied will
     try to guess based on the smallest difference in the data.
    :return: A dataframe where implicit missings are converted to explicit missings, but the min/max time for each
     group is preserved.
    """
    import pandas as pd

    if time_colname is None:
        for col in ('datetime', 'date', 'timestamp', 'time', 'dt'):
            if col in data.columns:
                time_colname = col
                break
        if time_colname is None:
            raise ValueError("Unable to guess `time_colname`, please pass")
    if dt_unit is None:
        diffs = data[time_colname].drop_duplicates().sort_values().diff()
        dt_unit = diffs.min()
        if dt_unit != diffs.value_counts().index[0]:
            raise ValueError("Unable to guess dt_unit, please pass")

    df_grid = pd.DataFrame(
        {time_colname: pd.date_range(data[time_colname].min(), data[time_colname].max(), freq=dt_unit)}
    )

    df_group_summary = data. \
        groupby(group_colname). \
        agg(_min=(time_colname, 'min'),
            _max=(time_colname, 'max')). \
        reset_index()

    # cross-join for all times to all groups (todo: not very memory efficient)
    df_cj = df_grid. \
        assign(_cj=1). \
        merge(df_group_summary.assign(_cj=1), how='left', on=['_cj'])
    # filter to min/max for each group
    df_cj = df_cj. \
        loc[df_cj[time_colname].between(df_cj['_min'], df_cj['_max']), [group_colname, time_colname]]. \
        reset_index(drop=True)
    return df_cj.merge(data, how='left', on=[group_colname, time_colname])


def chunk_grouped_data(*tensors, group_ids: Sequence) -> Sequence[Tuple[Tensor]]:
    """
    much faster approach to chunking than something like ``[X[gid==group_ids] for gid in np.unique(group_ids)]``
    """
    group_ids = np.asanyarray(group_ids)

    # torch.split requires we put groups into contiguous chunks:
    sort_idx = np.argsort(group_ids)
    group_ids = group_ids[sort_idx]
    tensors = [x[sort_idx] for x in tensors]

    # much faster approach to chunking than something like `[X[gid==group_ids] for gid in np.unique(group_ids)]`:
    _, counts_per_group = np.unique(group_ids, return_counts=True)
    counts_per_group = counts_per_group.tolist()

    group_data = []
    for chunk_tensors in zip(*(torch.split(x, counts_per_group) for x in tensors)):
        group_data.append(chunk_tensors)
    return group_data
