from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions


class HandlerBase:
    """
    Base-class for Handlers to provide the boiler plate to
    make them usable in context managers by provding stubs of
    ``__enter__``, ``__exit__`` and ``close``
    """

    specs = set()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
