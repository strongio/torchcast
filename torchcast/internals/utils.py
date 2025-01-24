import functools
from typing import Union, Any, Tuple, Sequence, List, Optional, Iterable

import torch

import numpy as np


def update_tensor(orig: torch.Tensor, new: torch.Tensor, mask: Optional[Union[slice, torch.Tensor]]) -> torch.Tensor:
    """
    In some cases we will want to save compute by only performing operations on a subset of a tensor, leaving the other
    elements as-is. In this case we'd need:

    .. code-block:: python

        new = orig.clone()
        new[mask] = some_op(orig[mask])

    However, in other cases we don't want to mask. If the second case is common, we're going to waste compute with the
    call to .clone().

    This function is a convenience function that will handle the masking for you, but only if a non-trivial mask is
    provided. If the mask is all True, it will return the new tensor as-is.

    :param orig: The original tensor.
    :param new: The new tensor. Should have the same shape as ``orig[mask]``.
    :param mask: A boolean mask Tensor.
    :return: If ``mask`` is all True, returns ``new`` (not a copy). Otherwise, returns a new tensor with the same shape
     as ``orig`` where the elements in ``mask`` are replaced with the elements in ``new``.
    """
    if isinstance(mask, slice) and not mask.start and not mask.stop and not mask.step:
        mask = None
    if mask is None or mask.all():
        return new
    else:
        out = orig.clone()
        out[mask] = new
        return out


def transpose_last_dims(x: torch.Tensor) -> torch.Tensor:
    args = list(range(len(x.shape)))
    args[-2], args[-1] = args[-1], args[-2]
    return x.permute(*args)


def get_nan_groups(isnan: torch.Tensor) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
    """
    Iterable of (group_idx, valid_idx) tuples that can be passed to torch.meshgrid. If no valid, then not returned; if
    all valid then (group_idx, None) is returned; can skip call to meshgrid.
    """
    assert len(isnan.shape) == 2
    state_dim = isnan.shape[-1]
    out: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
    if state_dim == 1:
        # shortcut for univariate
        group_idx = (~isnan.squeeze(-1)).nonzero().view(-1)
        out.append((group_idx, None))
        return out
    for nan_combo in torch.unique(isnan, dim=0):
        num_nan = nan_combo.sum()
        if num_nan < state_dim:
            c1 = (isnan * nan_combo[None, :]).sum(1) == num_nan
            c2 = (~isnan * ~nan_combo[None, :]).sum(1) == (state_dim - num_nan)
            group_idx = (c1 & c2).nonzero().view(-1)
            if num_nan == 0:
                valid_idx = None
            else:
                valid_idx = (~nan_combo).nonzero().view(-1)
            out.append((group_idx, valid_idx))
    return out


def get_owned_kwargs(module, kwargs: dict) -> Iterable[Tuple[str, str, Optional[torch.Tensor]]]:
    """
    Get keyword-arguments belonging to a module from a dictionary of kwargs passed to a multi-module container.

    :param module: Any object with an ``id`` and ``expected_kwargs``.
    :param kwargs: A dictionary of keyword arguments that are shared.
    :return: An iterable of tuples: ``(used_key, key_name, value)``. The first is used for indicating what was used
     (so at the end we can warn about unused keys), the second will be passed to the ``module`` later, and the last is
     the value that will be passed to the module.
    """
    if module.expected_kwargs is None:
        expected_kwargs = []
    else:
        expected_kwargs = module.expected_kwargs
    for k in kwargs:
        if k in expected_kwargs:
            yield k, k, kwargs[k]
        elif k.startswith(f'{module.id}__'):
            owner, _, subkey = k.partition("__")
            if subkey in expected_kwargs:
                yield k, subkey, kwargs[k]
            else:
                raise ValueError(
                    f"Found {k}, but {module.id} wasn't expecting a kwarg named '{subkey}'; expected:{expected_kwargs}"
                )


def validate_gt_shape(
        tensor: torch.Tensor,
        num_groups: int,
        num_times: int,
        trailing_dim: List[int]
) -> torch.Tensor:
    """
    Given we expect a tensor whose batch dimensions are (group, time) and with trailing dimensions, validate and
    standardize an input tensor. For validation, we check the shapes match the expected shapes. For standardization,
    we use the rules (1) if neither dims are present for num_groups or num_times, insert these dims and expand these,
    (2) if one more dim than len(trailing_dim) is present, assume it is the group dim, and expand the time dim.

    :param tensor: A tensor.
    :param num_groups: The number of group.
    :param num_times: The number of times.
    :param trailing_dim: Tuple with ints for trailing dim shape.
    :return: A tensor with shape (num_groups, num_times, *trailing_dim).
    """
    trailing_dim = list(trailing_dim)
    ntrailing = len(trailing_dim)

    if list(tensor.shape[-ntrailing:]) != trailing_dim:
        if ntrailing == 1 and tensor.shape[-1] == 1:
            # if input has singleton trailing dim, expand to match expected `trailing_dim`
            tensor = tensor.expand(torch.Size(list(tensor.shape[:-ntrailing]) + trailing_dim))
        else:
            raise ValueError(f"Expected `x.shape[-{ntrailing}:]` to be {trailing_dim}, got {tensor.shape[-ntrailing:]}")
    ndim = len(tensor.shape)
    if ndim == ntrailing:
        # insert dims for group and time:
        tensor = tensor.expand(torch.Size([num_groups, num_times] + trailing_dim))
    elif ndim == ntrailing + 1:
        # if we're only missing one dim and the first and last dims match, assume the time dim is singleton.
        if tensor.shape[0] == num_groups:
            tensor = tensor.unsqueeze(1).expand(torch.Size([-1, num_times] + trailing_dim))
        else:
            raise ValueError(f"Expected `x.shape[0]` to be ngroups, got {tensor.shape[0]}")
    elif ndim == ntrailing + 2:
        # note, does not allow singleton
        if tensor.shape[0] != num_groups or tensor.shape[1] != num_times:
            raise ValueError(f"Expected `x.shape[0:2]` to be (ngroups, ntimes), got {tensor.shape[0:2]}")
    else:
        raise ValueError(f"Expected len(x.shape) to be {ntrailing + 2} or {ntrailing}, got {ndim}")
    return tensor


def zpad(x: Any, n: int) -> str:
    return str(x).rjust(n, "0")


def identity(x: Any) -> Any:
    return x


def ragged_cat(tensors: Sequence[torch.Tensor],
               ragged_dim: int,
               cat_dim: int = 0,
               padding: Optional[float] = None) -> torch.Tensor:
    max_dim_len = max(tensor.shape[ragged_dim] for tensor in tensors)
    if padding is None:
        padding = float('nan')
    out = []
    num_dims = len(tensors[0].shape)
    for tensor in tensors:
        this_tens_dim_len = tensor.shape[ragged_dim]
        shape = list(tensor.shape)
        assert len(shape) == num_dims
        shape[ragged_dim] = max_dim_len
        padded = torch.empty(shape, dtype=tensors[0].dtype, device=tensors[0].device)
        padded[:] = padding
        idx = tuple(slice(0, this_tens_dim_len) if i == ragged_dim else slice(None) for i in range(num_dims))
        padded[idx] = tensor
        out.append(padded)
    return torch.cat(out, cat_dim)


@torch.no_grad()
def true1d_idx(arr: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
    if not isinstance(arr, torch.Tensor):
        arr = torch.as_tensor(arr)
    arr = arr.bool()
    if len(arr.shape) > 1:
        raise ValueError("Expected 1d array.")
    return arr.nonzero(as_tuple=True)[0]


def is_near_zero(tens: torch.Tensor, rtol: float = 1e-05, atol: float = 1e-08, equal_nan: bool = False) -> torch.Tensor:
    z = torch.zeros(1, dtype=tens.dtype, device=tens.device)
    return torch.isclose(tens, other=z, rtol=rtol, atol=atol, equal_nan=equal_nan)


def repeat(x: Union[torch.Tensor, np.ndarray], times: int, dim: int) -> Union[torch.Tensor, np.ndarray]:
    reps = [1 for _ in x.shape]
    reps[dim] = times
    if isinstance(x, np.ndarray):
        return np.tile(x, reps=reps)
    else:
        return x.repeat(*reps)


def chunk_grouped_data(*tensors, group_ids: Sequence) -> Sequence[Tuple[torch.Tensor]]:
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


class class_or_instancemethod:
    def __init__(self, method):
        self.method = method

    def __get__(self, instance, owner):
        @functools.wraps(self.method)
        def wrapped_method(*args, **kwargs):
            if instance is not None:
                return self.method(instance, *args, **kwargs)
            else:
                return self.method(owner, *args, **kwargs)

        return wrapped_method
