"""
This code was originally developed in https://github.com/NSLS-II/eiger-io
and copied (the lazy way, without retaining git history) into this repo.
It has been substantially changed from the original.
"""
from glob import glob
import os
from pathlib import Path

import dask.array
import h5py

from . import HandlerBase


class EigerHandler(HandlerBase):
    EIGER_MD_LAYOUT = {
        'y_pixel_size': 'entry/instrument/detector/y_pixel_size',
        'x_pixel_size': 'entry/instrument/detector/x_pixel_size',
        'detector_distance': 'entry/instrument/detector/detector_distance',
        'incident_wavelength': 'entry/instrument/beam/incident_wavelength',
        'frame_time': 'entry/instrument/detector/frame_time',
        'beam_center_x': 'entry/instrument/detector/beam_center_x',
        'beam_center_y': 'entry/instrument/detector/beam_center_y',
        'count_time': 'entry/instrument/detector/count_time',
        'pixel_mask': 'entry/instrument/detector/detectorSpecific/pixel_mask',
    }
    specs = {'AD_EIGER2', 'AD_EIGER'}

    def __init__(self, fpath, images_per_file=None, frame_per_point=None):
        '''
        Initializer for Eiger handler.

        Parameters
        ----------
        fpath: str
            the partial file path

        images_per_file: int, optional
            images per file. If not set, must set frame_per_point

        frame_per_point: int, optional. If not set, must set
            images_per_file

        This one is backwards compatible for both versions of resources
        saved in databroker. Old resources used 'frame_per_point' as a
        kwarg. Newer resources call this 'images_per_file'.
        '''
        self._file_prefix = fpath
        if images_per_file is None and frame_per_point is None:
            raise ValueError(
                "Either images_per_file or frame_per_point must be not None.")

        if images_per_file is None:
            # then grab from frame_per_point
            images_per_file = frame_per_point
        self._images_per_file = images_per_file
        self._files = {}

    def __call__(self, seq_id, frame_num=None):
        '''
        This returns data contained in the file.

        Parameters
        ----------
        seq_id: int
            The sequence id of the data

        frame_num: int or None
            If not None, return the frame_num'th image from this
            3D array. Useful for when an event is one image rather
            than a stack. (Editor's note: It's not clear what this original
            docstring was supposed to mean. Is it *ever* not None?)

        Returns
        -------
            A dask array
        '''
        master_path = Path(f'{self._file_prefix}_{seq_id}_master.h5').absolute()
        try:
            file = self._files[master_path]
        except KeyError:
            file = h5py.File(master_path, 'r')
            self._files[master_path] = file

        # TODO This should be captured in documents, not extracted here.
        # This code is retained just in case, but it does not do anything other
        # than set self._md, which has no effect unless the user obtain direct
        # access to the handler instance by reaching into the Filler's cache of
        # them.
        md = {k: file[v][()] for k, v in self.EIGER_MD_LAYOUT.items()}
        # the pixel mask from the eiger contains:
        # 1  -- gap
        # 2  -- dead
        # 4  -- under-responsive
        # 8  -- over-responsive
        # 16 -- noisy
        pixel_mask = md['pixel_mask']
        # pixel_mask[pixel_mask>0] = 1
        # pixel_mask[pixel_mask==0] = 2
        # pixel_mask[pixel_mask==1] = 0
        # pixel_mask[pixel_mask==2] = 1
        md['binary_mask'] = (pixel_mask == 0)
        md['framerate'] = 1. / float(md['frame_time'])
        self._md = md

        try:
            # Eiger firmware v1.3.0 and onwards
            entry = file['entry']['data']
        except KeyError:
            # Older firmwares
            entry = file['entry']

        # Each 'master' file references multiple 'data' files.
        # We just need to know many there are, but here we make
        # a sorted list of their names because it can be handy
        # when things break and we need to debug.
        data_files = sorted([key for key in entry.keys() if key.startswith("data")])

        to_concatenate = []
        for i in range(len(data_files)):
            dataset = entry[f'data_{1 + (i // self._images_per_file):06d}']
            da = dask.array.from_array(dataset)
            to_concatenate.append(da)
        stack = dask.array.concatenate(to_concatenate)
        if frame_num is None:
            return stack
        else:
            return stack[frame_num % self.images_per_file]

    def get_file_list(self, datum_kwargs_gen):
        '''
        Get the file list.

        Receives a list of datum_kwargs for each datum.
        '''
        filenames = []
        for dm_kw in datum_kwargs_gen:
            seq_id = dm_kw['seq_id']
            new_filenames = glob(f'{self._file_prefix}_{seq_id}*')
            filenames.extend(new_filenames)

        return filenames

    def get_file_sizes(self, datum_kwargs_gen):
        '''
        Get the file size.

        Returns size in bytes.
        '''
        sizes = []
        file_name = self.get_file_list(datum_kwargs_gen)
        for file in file_name:
            sizes.append(os.path.getsize(file))

        return sizes

    def close(self):
        for file in self._files.values():
            file.close()
