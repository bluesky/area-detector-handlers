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
        """
        Subclasses should clean up all resources here.

        This includes open files, network connections, and internal memory allocations.
        """
        pass

    def __del__(self):
        # Intentionally do NOT close() on __del__.

        # It is common for handlers to go out of scope and be garbage collected
        # wnile there are still objects wrapping the handlers' open file(s)
        # with deferred I/O yet to be performed. If we close the files as part
        # of handler garbage collection, the deferred I/O will fail when it
        # tries to operate on a closed file.

        # For example, HDF5 handlers return dask.arrays wrapping h5py Datasets.
        # The corresponding h5py.File must still be open when those dask arrays
        # are computed. It is not unusual for a BlueskyRun and its handlers to
        # go out of scope and be garbage collected before the arrays are
        # computed.

        # Explicitly closing the handler with close() or using it as a context
        # is a different story. In that case the calling code is declaring that
        # it is explictly done with any associated I/O.
        pass
