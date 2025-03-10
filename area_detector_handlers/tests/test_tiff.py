from area_detector_handlers.tests.conftest import select_handler
import numpy as np


@select_handler("AD_TIFF")
def test_tiff(tiff_files, handler):
    (rpath, kwargs), (N_rows, N_cols, N_points, fpp) = tiff_files
    expected_shape = (fpp, N_rows, N_cols)
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            assert d.shape == expected_shape
            assert np.all(d == frame)


@select_handler("AD_TIFF_ND_TS")
def test_tiff_nd_ts(tiff_files, handler):
    (rpath, kwargs), (_, _, N_points, fpp) = tiff_files
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            assert d.shape == (fpp,)
            assert np.all(d == np.linspace(frame * fpp, (frame + 1) * fpp - 1, fpp))


@select_handler("AD_TIFF_TS")
def test_tiff_ts(tiff_files, handler):
    (rpath, kwargs), (_, _, N_points, fpp) = tiff_files
    with handler(rpath, **kwargs) as h:
        for frame in range(N_points):
            d = h(point_number=frame)
            assert d.shape == (fpp,)
            expected = np.linspace(frame * fpp, (frame + 1) * fpp - 1, fpp) + np.linspace(0, fpp - 1, fpp) * 1e-9
            assert np.all(d == expected)
