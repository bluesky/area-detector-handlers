import h5py
import numpy as np

from area_detector_handlers.tests.conftest import select_handler


@select_handler("AD_EIGER")
def test_eiger(eigerfile, handler):
    (out_name, kwargs), (N_points, N_chans, N_bin, N_roi) = eigerfile
    with h5py.File(out_name, 'r') as file:
        ndarr = file['entry/data/data_000001'][()]

    fpath, seq_id = out_name.split('_')[0:2]
    with handler(fpath, **kwargs) as h:
        dask_array = h(seq_id)
        assert np.array_equal(dask_array.compute(), ndarr)


@select_handler("AD_EIGER2")
def test_eiger2(eigerfile, handler):
    (out_name, kwargs), (N_points, N_chans, N_bin, N_roi) = eigerfile
    with h5py.File(out_name, 'r') as file:
        ndarr = file['entry/data/data_000001'][()]

    fpath, seq_id = out_name.split('_')[0:2]
    with handler(fpath, **kwargs) as h:
        dask_array = h(seq_id)
        assert np.array_equal(dask_array.compute(), ndarr)
