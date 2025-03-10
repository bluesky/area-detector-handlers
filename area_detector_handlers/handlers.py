import logging
import os.path
import struct
from typing import cast

import dask
import dask.array
import h5py
import numpy as np
import pandas as pd
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
        return np.array(ret)

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class AreaDetectorTiffTimestampHandler(AreaDetectorTiffHandler):
    """
    Handler to retrieve timestamps from AreaDetector TIFF files.

    The timestamps are read from the TIFF file's EPICS metadata.
    """

    specs = {"AD_TIFF_TS"} | AreaDetectorTiffHandler.specs

    def __call__(self, point_number):
        ret = []
        for fn in self._fnames_for_point(point_number):
            with tifffile.TiffFile(fn) as tif:
                if tif.epics_metadata is None:
                    raise ValueError("TIFF file has no EPICS metadata, "
                                     "was this file written by the area "
                                     "detector plugin?")
                ret.append(tif.epics_metadata["timeStamp"])
        return np.array(ret)


H5PY_KEYERROR_IOERROR_MSG = (
    "h5py raised a KeyError but this can sometimes actually "
    "mean an IOError. Raising as an IOError so that Filler will "
    "retry. (After several retries with backoff Filler gives up.)")


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
        if self._dataset is None:
            try:
                self._dataset = self._file[self._key]
            except KeyError as error:
                raise IOError(H5PY_KEYERROR_IOERROR_MSG) from error
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
        if self._dataset is None:
            try:
                self._dataset = dask.array.from_array(self._file[self._key])
            except KeyError as error:
                raise IOError(H5PY_KEYERROR_IOERROR_MSG) from error
        start = point_number * self._fpp
        stop = (point_number + 1) * self._fpp
        if not len(self._dataset[start:stop]):
            raise ValueError('Invalid slicing bounds. Handler is slicing beyond size of dataset')
        return self._dataset[start:stop]


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

    EPICS timestamps are read from the HDF5 file's NDArrayEpicsTSSec and NDArrayEpicsTSnSec attributes.
    They are typically in seconds relative to 1990-01-01 00:00:00.

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

    def get_file_list(self, datum_kwarg_gen):
        return [self._filename]

    def __call__(self, point_number):
        # Don't read out the dataset until it is requested for the first time.
        if not self._dataset1:
            try:
                self._dataset1 = self._file[self._key[0]]
            except KeyError as error:
                raise IOError(H5PY_KEYERROR_IOERROR_MSG) from error
        if not self._dataset2:
            try:
                self._dataset2 = self._file[self._key[1]]
            except KeyError as error:
                raise IOError(H5PY_KEYERROR_IOERROR_MSG) from error
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


class AreaDetectorHDF5NDArrayTimestampHandler(AreaDetectorHDF5TimestampHandler):
    """ Handler to retrieve timestamps from Areadetector HDF5 File

    In this spec, the timestamps of the images are read.
    The timestamps are returned as a numpy array.

    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """

    specs = {"AD_HDF5_NDARRAY_TS"} | AreaDetectorHDF5TimestampHandler.specs

    def __init__(self, filename, frame_per_point=1):
        super().__init__(filename, frame_per_point)
        self._key = [
            "/entry/instrument/NDAttributes/NDArrayTimeStamp",
        ]
        self._timestamps = None

    def __call__(self, point_number):
        if not self._timestamps:
            if self._file is None:
                self.open()
            file = cast(h5py.File, self._file)
            try:
                self._timestamps = file[self._key[0]]
            except KeyError as error:
                raise IOError(H5PY_KEYERROR_IOERROR_MSG) from error
        start, stop = point_number * self._fpp, (point_number + 1) * self._fpp
        rtn = self._timestamps[start:stop].squeeze()
        return rtn


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


def read_imm_header(file):
    imm_headformat = "ii32s16si16siiiiiiiiiiiiiddiiIiiI40sf40sf40sf40sf40sf40sf40sf40sf40sf40sfffiiifc295s84s12s"
    imm_fieldnames = [
        'mode',
        'compression',
        'date',
        'prefix',
        'number',
        'suffix',
        'monitor',
        'shutter',
        'row_beg',
        'row_end',
        'col_beg',
        'col_end',
        'row_bin',
        'col_bin',
        'rows',
        'cols',
        'bytes',
        'kinetics',
        'kinwinsize',
        'elapsed',
        'preset',
        'topup',
        'inject',
        'dlen',
        'roi_number',
        'buffer_number',
        'systick',
        'pv1',
        'pv1VAL',
        'pv2',
        'pv2VAL',
        'pv3',
        'pv3VAL',
        'pv4',
        'pv4VAL',
        'pv5',
        'pv5VAL',
        'pv6',
        'pv6VAL',
        'pv7',
        'pv7VAL',
        'pv8',
        'pv8VAL',
        'pv9',
        'pv9VAL',
        'pv10',
        'pv10VAL',
        'imageserver',
        'CPUspeed',
        'immversion',
        'corecotick',
        'cameratype',
        'threshhold',
        'byte632',
        'empty_space',
        'ZZZZ',
        'FFFF'
    ]
    bindata = file.read(1024)

    imm_headerdat = struct.unpack(imm_headformat, bindata)
    imm_header = dict(zip(imm_fieldnames, imm_headerdat))

    return imm_header


class IMMHandler(HandlerBase):
    """
    Handler to retrieve data from the IMM format.

    Based on the following:
    - https://github.com/AdvancedPhotonSource/xpcs-eigen/blob/master/python/io/imm_file.py
    - https://pypi.org/project/pyimm

    Parameters
    ----------
    filename: string
        path to .imm file
    frames_per_point: integer
        number of frames to return as one datum
    """
    specs = {"IMM"} | HandlerBase.specs
    return_type = {'delayed': True}

    def __init__(self, filename, frames_per_point):
        self.filename = filename
        with open(filename, "rb") as file:
            self.frames_per_point = frames_per_point
            header = read_imm_header(file)
            self.rows, self.cols = header['rows'], header['cols']
            self.is_compressed = bool(header['compression'] == 6)
            file.seek(0)
            self.toc = []  # (start byte, element count) pairs
            while True:
                try:
                    header = read_imm_header(file)
                    cur = file.tell()
                    payload_size = header['dlen'] * (6 if self.is_compressed else 2)
                    self.toc.append((cur, header['dlen']))
                    file_pos = payload_size + cur
                    file.seek(file_pos)
                    # Check for end of file.
                    if not file.peek(4):
                        break
                except Exception as err:
                    raise IOError("IMM file doesn't seems to be of right type") from err

    def close(self):
        # This handler does not cache any file handles.
        pass

    def __call__(self, index):

        shape = (1, self.rows, self.cols)

        @dask.delayed
        def load_plane(j):
            # Load plane 'j' inside the chunk correspond to the Datum
            # identified by 'index'.
            with open(self.filename, "rb") as file:
                start_byte, num_pixels = self.toc[index * self.frames_per_point + j]
                file.seek(start_byte)
                indexes = np.fromfile(file, dtype=np.uint32, count=num_pixels)
                values = np.fromfile(file, dtype=np.uint16, count=num_pixels)
            # TODO Here is where we would use pydata sparse instead of literal
            # numpy.
            # Start with a zeroed array.
            result = np.zeros((self.rows * self.cols), np.uint32)
            # Fill in the sparse data.
            result[indexes] = values
            # Fix the shape.
            result_reshaped = result.reshape(*shape)
            return result_reshaped

        chunks = []
        for j in range(self.frames_per_point):
            delayed_arr = dask.array.from_delayed(
                load_plane(j), shape=shape, dtype=np.uint32)
            chunks.append(delayed_arr)

        result = dask.array.concatenate(chunks, axis=0)
        return result

    def get_file_list(self, datum_kwargs_gen):
        return [self.filename]


class HDF5SingleHandler(HandlerBase):
    """
    Handler for hdf5 data stored 1 image per file.

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    key : string
        the 'path' inside the file to the data set.
    frame_per_point : float
        the number of frames per point.
    """
    specs = {'AD_HDF5_SINGLE'}  # Used by SIX

    def __init__(self, fpath, template, filename, key, frame_per_point=1):
        self._path = os.path.join(fpath, '')
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        ret = []
        for fn in self._fnames_for_point(point_number):
            f = h5py.File(fn, 'r')
            data = f[self._key][:]
            ret.append(data)
        return np.stack(ret)

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class AreaDetectorHDF5SingleHandler(HDF5SingleHandler):
    """
    Handler for hdf5 data stored 1 image per file by areadetector

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    frame_per_point : float
        the number of frames per point.
    """
    def __init__(self, fpath, template, filename, frame_per_point=1):
        hardcoded_key = '/entry/data/data'
        super(AreaDetectorHDF5SingleHandler, self).__init__(
            fpath=fpath, template=template, filename=filename,
            key=hardcoded_key, frame_per_point=frame_per_point)


class SpecsHDF5SingleHandlerDataFrame(HandlerBase):
    """Handler for hdf5 data stored 1 image per file and returned as a
-    Pandas.DataFrame.

    This will work with all hdf5 files that are a mxn arrays and the data is
    'table like' where m is the number of columns and n is the number of rows.

    Parameters
    ----------
    fpath : string
        filepath
    template : string
        filename template string.
    filename : string
        filename
    key : string
        the 'path' inside the file to the data set.
    column_names : list[str]
        The column names of the table
    frame_per_point : float
        the number of frames per point.
    """
    specs = {'SPECS_HDF5_SINGLE_DATAFRAME'}  # Used by IOS

    def __init__(self, fpath, template, filename, key='/entry/data/data',
                 column_names=None, frame_per_point=1):
        # I have included defaults for `key` and 'column_names' for back
        # compatibility with existing files at SIX.
        self._path = os.path.join(fpath, '')
        self._fpp = frame_per_point
        self._template = template
        self._filename = filename
        self._key = key
        self._column_names = column_names

    def _fnames_for_point(self, point_number):
        start = int(point_number * self._fpp)
        stop = int((point_number + 1) * self._fpp)
        for j in range(start, stop):
            yield self._template % (self._path, self._filename, j)

    def __call__(self, point_number):
        ret = []
        for fn in self._fnames_for_point(point_number):
            with h5py.File(fn, 'r') as f:
                dataframe = pd.DataFrame(np.array(f[self._key][:]).transpose(),
                                         columns=self._column_names)
                start_energy = f['/entry/instrument/NDAttributes/StartEnergy'][0]
                stop_energy = f['/entry/instrument/NDAttributes/StopEnergy'][0]
                step_energy = f['/entry/instrument/NDAttributes/StepEnergy'][0]
                kinetic_energy = np.append(
                    np.arange(start_energy, stop_energy, step_energy), stop_energy)
                dataframe['kinetic_energy'] = kinetic_energy
            ret.append(dataframe)
        return ret

    def get_file_list(self, datum_kwargs):
        ret = []
        for d_kw in datum_kwargs:
            ret.extend(self._fnames_for_point(**d_kw))
        return ret


class TimepixHDF5Handler(HDF5DatasetSliceHandler):
    """
    Handler for the 'AD_HDF5' spec used by Area Detectors.
    In this spec, the key (i.e., HDF5 dataset path) is always
    '/entry/detector/data'.
    Parameters
    ----------
    filename : string
        path to HDF5 file
    frame_per_point : integer, optional
        number of frames to return as one datum, default 1
    """
    _handler_name = 'TPX_HDF5'
    specs = {_handler_name}

    # Ported from
    #
    # https://github.com/NSLS-II-HXN/hxntools/blob/16bcf9b16e962e2ba8bda6bc7dc56694482c3af3/
    # hxntools/handlers/timepix.py#L56-L77
    #
    # TODO this is only different due to the hardcoded key being different?
    hardcoded_key = '/entry/instrument/detector/data'

    def __init__(self, filename, frame_per_point=1):
        super(TimepixHDF5Handler, self).__init__(
                filename=filename, key=self.hardcoded_key,
                frame_per_point=frame_per_point)
