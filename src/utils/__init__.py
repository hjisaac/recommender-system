class cached_class_property:  # noqa
    """
    A decorator to define a class-level property that caches its value
    upon first access, avoiding re-computation on subsequent calls.
    """

    def __init__(self, func):
        self.func = func
        self._cache_name = f"_{func.__name__}_cache"

    def __get__(self, instance, cls):
        if not hasattr(cls, self._cache_name):
            setattr(cls, self._cache_name, self.func(cls))
        return getattr(cls, self._cache_name)
