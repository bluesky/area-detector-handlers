import logging
import os.path

import dask.array
import h5py
import numpy as np
import tifffile

from . import HandlerBase
from .spe_reader import PrincetonSPEFile

from ._xspress3 import BulkXSPRESS, Xspress3HDF5Handler  # noqa

logger = logging.getLogger(__name__)


class IntegrityError(Exception):
    pass


class AreaDetectorSPEHandler(HandlerBase):
    specs = {"AD_SPE"} | HandlerBase.specs

    def __init__(self, fpath, template, filename, frame_per_point=1):
        self._path = os.path.join(fpath, "")
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._f_cache = dict()

    def __call__(self, point_number):
        if point_number not in self._f_cache:
            fname = self._template % (self._path, self._filename, point_number)
            spe_obj = PrincetonSPEFile(fname)
            self._f_cache[point_number] = spe_obj

        spe = self._f_cache[point_number]
        data = spe.getData()

        if data.shape[0] != self._fpp:
            raise IntegrityError(
                "expected {} frames, found {} frames".format(self._fpp, data.shape[0])
            )
        return data.squeeze()

    def get_file_list(self, datum_kwarg_gen):
        return [
            self._template % (self._path, self._filename, d["point_number"])
            for d in datum_kwarg_gen
        ]


class AreaDetectorTiffHandler(HandlerBase):
    specs = {"AD_TIFF"} | HandlerBase.specs

    def __init__(self, fpath, template, filename, frame_per_point=1):
        self._path = os.path.join(fpath, "")
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        ret = []
        for fn in self._fnames_for_point(point_number):
            with tifffile.TiffFile(fn) as tif:
                ret.append(tif.asarray())
        return np.array(ret).squeeze()

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class HDF5DatasetSliceHandlerPureNumpy(HandlerBase):
    """
    Handler for data stored in one Dataset of an HDF5 file.

    Parameters
    ----------
    filename : string
        path to HDF5 file
    key : string
        key of the single HDF5 Dataset used by this Handler
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    swmr : bool, optional
        Open the hdf5 file in SWMR read mode. Only used when mode = 'r'.
        Default is False.
    """
    return_type = {'delayed': False}
    specs = set()

    def __init__(self, filename, key, frame_per_point=1):
        self._fpp = frame_per_point
        self._filename = filename
        self._key = key
        self._file = None
        self._dataset = None
        self._data_objects = {}
        self.open()

    def get_file_list(self, datum_kwarg_gen):
        return [self._filename]

    def __call__(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        if not self._dataset:
            self._dataset = self._file[self._key]
        start = point_number * self._fpp
        stop = (point_number + 1) * self._fpp
        return self._dataset[start:stop]

    def open(self):
        if self._file:
            return

        self._file = h5py.File(self._filename, "r")

    def close(self):
        super().close()
        self._file.close()
        self._file = None


class HDF5DatasetSliceHandler(HDF5DatasetSliceHandlerPureNumpy):
    return_type = {'delayed': True}

    def __call__(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        if not self._dataset:
            self._dataset = self._file[self._key]
        start = point_number * self._fpp
        stop = (point_number + 1) * self._fpp
        return dask.array.from_array(self._dataset)[start:stop]


class AreaDetectorHDF5Handler(HDF5DatasetSliceHandler):
    """
    Handler for the 'AD_HDF5' spec used by Area Detectors.

    In this spec, the key (i.e., HDF5 dataset path) is always
    '/entry/data/data'.

    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """

    specs = {"AD_HDF5"} | HDF5DatasetSliceHandler.specs

    def __init__(self, filename, frame_per_point=1):
        hardcoded_key = "/entry/data/data"
        super().__init__(
            filename=filename, key=hardcoded_key, frame_per_point=frame_per_point
        )


class AreaDetectorHDF5SWMRHandler(AreaDetectorHDF5Handler):
    """
    Handler for the 'AD_HDF5_SWMR' spec used by Area Detectors.

    In this spec, the key (i.e., HDF5 dataset path) is always
    '/entry/data/data'.

    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """

    specs = {"AD_HDF5_SWMR"} | HDF5DatasetSliceHandler.specs

    def open(self):
        if self._file:
            return

        self._file = h5py.File(self._filename, "r", swmr=True)

    def __call__(self, point_number):
        if self._dataset is not None:
            self._dataset.id.refresh()
        rtn = super().__call__(point_number)

        return rtn


class AreaDetectorHDF5TimestampHandler(HandlerBase):
    """ Handler to retrieve timestamps from Areadetector HDF5 File

    In this spec, the timestamps of the images are read.

    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """

    specs = {"AD_HDF5_TS"} | HandlerBase.specs

    def __init__(self, filename, frame_per_point=1):
        self._fpp = frame_per_point
        self._filename = filename
        self._key = [
            "/entry/instrument/NDAttributes/NDArrayEpicsTSSec",
            "/entry/instrument/NDAttributes/NDArrayEpicsTSnSec",
        ]
        self._file = None
        self._dataset1 = None
        self._dataset2 = None
        self.open()

    def __call__(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        if not self._dataset1:
            self._dataset1 = self._file[self._key[0]]
        if not self._dataset2:
            self._dataset2 = self._file[self._key[1]]
        start, stop = point_number * self._fpp, (point_number + 1) * self._fpp
        rtn = self._dataset1[start:stop].squeeze()
        rtn = rtn + (self._dataset2[start:stop].squeeze() * 1e-9)
        return rtn

    def open(self):
        if self._file:
            return
        self._file = h5py.File(self._filename, "r")

    def close(self):
        super().close()
        self._file.close()
        self._file = None


class AreaDetectorHDF5SWMRTimestampHandler(AreaDetectorHDF5TimestampHandler):
    """ Handler to retrieve timestamps from Areadetector HDF5 File

    In this spec, the timestamps of the images are read. Reading
    is done using SWMR option to allow read during processing

    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """

    specs = {"AD_HDF5_SWMR_TS"} | HandlerBase.specs

    def open(self):
        if self._file:
            return
        self._file = h5py.File(self._filename, "r", swmr=True)

    def __call__(self, point_number):
        if (self._dataset1 is not None) and (self._dataset2 is not None):
            self._dataset.id.refresh()
        rtn = super().__call__(point_number)
        return rtn


class PilatusCBFHandler:
    specs = {"AD_CBF"}

    def __init__(self, rpath, template, filename, frame_per_point=1, initial_number=1):
        self._path = os.path.join(rpath, "")
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._initial_number = initial_number

    def __call__(self, point_number):
        # Delay import because this is believed to be rarely used and should
        # not be a required dependency.
        try:
            import fabio
        except ImportError as exc:
            raise ImportError(
                "The AreaDetectorSPEHandler handler requires fabio to be installed."
            ) from exc
        start, stop = (
            self._initial_number + point_number * self._fpp,
            (point_number + 2) * self._fpp,
        )
        ret = []
        # commented out by LY to test scan speed imperovement, 2017-01-24
        for j in range(start, stop):
            fn = self._template % (self._path, self._filename, j)
            img = fabio.open(fn)
            ret.append(img.data)
        return np.array(ret).squeeze()

    def get_file_list(self, datum_kwargs_gen):
        file_list = []
        for dk in datum_kwargs_gen:
            point_number = dk["point_number"]
            start, stop = (
                self._initial_number + point_number * self._fpp,
                (point_number + 2) * self._fpp,
            )
            for j in range(start, stop):
                fn = self._template % (self._path, self._filename, j)
                file_list.append(fn)
        return file_list
