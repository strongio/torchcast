"""
These are simple training classes for PyTorch models, with specialized subclasses for torchcast's model-classes (i.e.,
when the data are too big for the :func:`torchcast.state_space.StateSpaceModel.fit()` method). Additionaly, there is a
special class for training neural networks to embed complex seasonal patterns into lower dimensional embeddings.

While the classes in this module are helpful for quick development, they are not necessarily meant to replace more
sophisticated tools (e.g. PyTorch Lightning) in more complex settings.
"""
import warnings
from itertools import zip_longest

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Optimizer
import torch.nn as nn
from typing import Generator, Union, Type, Sequence, Tuple, Dict, Optional

from tqdm.auto import tqdm

from torchcast.process import LinearModel
from torchcast.state_space import Predictions
from torchcast.utils import TimeSeriesDataset
from torchcast.utils.features import fourier_model_mat


class BaseTrainer:
    _device = None

    def __init__(self,
                 module: nn.Module,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam):
        self.module = module
        if isinstance(optimizer, type):
            optimizer = optimizer([p for p in module.parameters() if p.requires_grad])
        self.optimizer = optimizer

        self.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.epoch = 0

    def to(self, device) -> 'BaseTrainer':
        self.module.to(device)
        self._device = device
        return self

    def _get_closure(self, batch: any) -> callable:
        raise NotImplementedError

    def _get_batch_numel(self, batch: any) -> int:
        raise NotImplementedError

    def __call__(self, dataloader: DataLoader, prog: bool = True) -> Generator[float, None, None]:
        if prog:
            prog = tqdm(total=len(dataloader))

        self.module.train()
        while True:
            self.epoch += 1
            epoch_loss = 0.
            n = 0
            try:
                for batch in dataloader:
                    closure = self._get_closure(batch)
                    loss = self.optimizer.step(closure)
                    batch_n = self._get_batch_numel(batch)
                    epoch_loss += (loss.item() * batch_n)
                    n += batch_n
                    if prog:
                        prog.update()
                        prog.set_description(f'Epoch {self.epoch}')
            except KeyboardInterrupt:
                if prog:
                    prog.close()
                break
            finally:
                prog.reset()

            yield epoch_loss / n

        if prog:
            prog.close()


class SimpleTrainer(BaseTrainer):
    """
    A simple trainer for a standard nn.Module (not a state-space model).
    """
    _warned = False

    def __init__(self,
                 module: nn.Module,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam,
                 loss_fn: callable = torch.nn.MSELoss()):
        self.loss_fn = loss_fn
        super().__init__(module=module, optimizer=optimizer)

    def _get_closure(self, batch: Dataset) -> callable:
        inputs, targets, *_other = batch
        if len(_other) and not self._warned:
            warnings.warn("Ignoring additional tensors in batch.")
            self._warned = True
        inputs, targets = inputs.to(self._device), targets.to(self._device)

        def closure():
            self.optimizer.zero_grad()
            outputs = self.module(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: Dataset) -> int:
        if getattr(self.loss_fn, 'reduction', 'mean') != 'mean':
            raise RuntimeError("Can't handle loss functions with `reduction` other than 'mean'")
        return len(batch)


class StateSpaceTrainer(BaseTrainer):
    """
    TODO

    :param module: A :class:`torchcast.state_space.StateSpaceModel` instance (e.g. ``KalmanFilter`` or ``ExpSmoother``).
    :param kwargs_getter: If a sequence of strings are passed, these map the ``tensors[1:]`` of each
     :class:`torchcast.utils.TimeSeriesDataset` to the corresponding keyword arguments of the module's ``forward``
     method. If left unspecified, will pass ``tensor[1]`` as ``X`` (if the batch has such a tensor). You can also pass
     a callable that takes the :class:`torchcast.utils.TimeSeriesDataset` and returns a dictionary that will be passed
     as forward-kwargs.
    :param optimizer: An optimizer or a class to instantiate an optimizer
    """

    def __init__(self,
                 module: nn.Module,
                 kwargs_getter: Optional[Sequence[str]] = None,
                 optimizer: Union[Optimizer, Type[Optimizer]] = torch.optim.Adam):

        self.kwargs_getter = kwargs_getter
        super().__init__(module=module, optimizer=optimizer)

    def get_loss(self, pred: Predictions, y: torch.Tensor) -> torch.Tensor:
        return -pred.log_prob(y).mean()

    def _batch_to_args(self, batch: TimeSeriesDataset) -> Tuple[torch.Tensor, dict]:
        y = batch.tensors[0]

        if callable(self.kwargs_getter):
            kwargs = self.kwargs_getter(batch)
        else:
            if self.kwargs_getter is None:
                self.kwargs_getter = ['X'] if len(batch.tensors) > 1 else []

            kwargs = {}
            for i, (k, t) in enumerate(zip_longest(self.kwargs_getter, batch.tensors[1:])):
                if k is None:
                    raise RuntimeError(
                        f"Found element-{i + 1} of the dataset.tensors, but `kwargs_getter` doesn't have enough "
                        f"elements: {self.kwargs_getter}"
                    )
                if t is None:
                    raise RuntimeError(
                        f"Found element-{i} of `kwargs_getter`, but `dataset.tensors` doesn't have enough "
                        f"elements: {batch}"
                    )
                kwargs[k] = t
        return y, kwargs

    def _get_closure(self, batch: TimeSeriesDataset) -> callable:

        def closure():
            # we call _batch_to_args from inside the closure in case `kwargs_getter` is callable & involves grad.
            # only scenario this would matter is if optimizer is LBFGS (or another custom optimizer that calls closure
            # multiple times per step), in which case grad from callable would be lost after the first step.
            y, kwargs = self._batch_to_args(batch)
            self.optimizer.zero_grad()
            pred = self.module(y, **kwargs)
            loss = self.get_loss(pred, y)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: any) -> int:
        return batch.tensors[0].numel()


class SeasonalEmbeddingsTrainer(BaseTrainer):
    """
    This trainer is designed to train a :class:`torch.nn.Module` with
    """

    # TODO: classmethod `from_dt_unit` or something that puts in decent defaults for hourly, daily, weekly data

    def __init__(self,
                 module: nn.Module,
                 yearly: int,
                 weekly: int,
                 daily: int,
                 other: Sequence[Tuple[np.timedelta64, int]] = (),
                 loss_fn: callable = torch.nn.MSELoss(),
                 **kwargs):
        super().__init__(module=module, **kwargs)
        self.weekly = weekly
        self.yearly = yearly
        self.daily = daily
        self.other = other
        self.loss_fn = loss_fn

        self._warned = False

    def times_to_arrays(self, times: np.ndarray) -> Dict[str, torch.Tensor]:
        out = {}
        if self.yearly:
            out['yearly'] = torch.as_tensor(fourier_model_mat(times, K=self.yearly, period='yearly'))
        if self.weekly:
            out['weekly'] = torch.as_tensor(fourier_model_mat(times, K=self.weekly, period='weekly'))
        if self.daily:
            out['daily'] = torch.as_tensor(fourier_model_mat(times, K=self.daily, period='daily'))
        for period, k in self.other:
            out[str(period)] = torch.as_tensor(fourier_model_mat(times, K=k, period=period))
        return out

    def times_to_model_mat(self, times: np.ndarray) -> torch.Tensor:
        arrays = list(self.times_to_arrays(times).values())
        return torch.cat(arrays, -1)

    def get_loss(self, predictions: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nan_mask = torch.isnan(y)
        return self.loss_fn(predictions[~nan_mask], y[~nan_mask])

    def _getXy(self, batch: TimeSeriesDataset) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = batch.to(self._device)

        y, *_other = batch.tensors
        X = self.times_to_model_mat(batch.times()).to(y.dtype)

        if len(_other) == 1:
            X = torch.cat([X, _other[0]], -1)
        elif len(_other) and not self._warned:
            warnings.warn("Ignoring additional tensors in batch.")
            self._warned = True
        return X, y

    def _get_closure(self, batch: TimeSeriesDataset) -> callable:
        X, y = self._getXy(batch)

        def closure():
            self.optimizer.zero_grad()
            emb = self.module(X)
            outputs = LinearModel.solve_and_predict(y=y, X=emb)  # handles nans
            loss = self.get_loss(outputs, y)
            loss.backward()
            return loss

        return closure

    def _get_batch_numel(self, batch: any) -> int:
        return batch.tensors[0].numel()

    def predict(self, batch: TimeSeriesDataset) -> torch.Tensor:
        """
        Since this requires passing `y` it's not really useful genuine prediction, but is primarily for
        visualizing/sanity-checking outputs after/during training.
        """
        X, y = self._getXy(batch)
        emb = self.module(X)
        return LinearModel.solve_and_predict(y=y, X=emb)
