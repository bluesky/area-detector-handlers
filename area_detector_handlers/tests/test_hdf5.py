from area_detector_handlers.tests.conftest import select_handler
import numpy as np

import pytest


@select_handler("AD_HDF5")
def test_hdf5(hdf5_files, handler):
    (rpath, kwargs), (N_rows, N_cols, N_points, fpp) = hdf5_files
    expected_shape = (fpp, N_rows, N_cols)
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            assert d.shape == expected_shape
            assert np.all(d == frame)


@select_handler("AD_HDF5")
def test_hdf5_slicing_bound_exception(hdf5_files, handler):
    (rpath, kwargs), (N_rows, N_cols, N_points, fpp) = hdf5_files
    bad_num_points = 10
    assert bad_num_points > N_points
    with pytest.raises(ValueError):
        with handler(rpath, **kwargs) as h:
            for frame in range(bad_num_points):
                h(point_number=frame)


@select_handler("AD_HDF5_ND_TS")
def test_hdf5_nd_ts(hdf5_files, handler):
    (rpath, kwargs), (_, _, N_points, fpp) = hdf5_files
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            if fpp == 1:
                assert d.shape == ()
                assert d == frame
            else:
                assert d.shape == (fpp,)
                assert np.all(d == np.linspace(frame * fpp, (frame + 1) * fpp - 1, fpp))


@select_handler("AD_HDF5_TS")
def test_hdf5_ts(hdf5_files, handler):
    (rpath, kwargs), (_, _, N_points, fpp) = hdf5_files
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            if fpp == 1:
                assert d.shape == ()
                assert d == frame + 1e-9
            else:
                assert d.shape == (fpp,)
                assert np.all(d == np.linspace(frame * fpp, (frame + 1) * fpp - 1, fpp) + 1e-9)
