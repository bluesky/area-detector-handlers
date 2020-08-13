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
