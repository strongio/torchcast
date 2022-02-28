import functools


def wrap_fit(fit: callable, new_tol: float, new_patience: int) -> callable:
    """
    Readthedocs limits build-time, so tweak fit default args.
    """

    @functools.wraps(fit)
    def new_fit(self, *args, **kwargs):
        return fit(self, *args, tol=new_tol, patience=new_patience, **kwargs)

    return new_fit
