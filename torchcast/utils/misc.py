import functools


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
