from area_detector_handlers.tests.conftest import select_handler


@select_handler("AD_EIGER")
def test_pre_pixel(eigerfile, handler):
    (fname, kwargs), (N_points, N_chans, N_bin, N_roi) = eigerfile

    with handler(fname, **kwargs) as h:
        for frame in range(0, N_points):
            for chan in range(1, N_chans + 1): 
                assert h(frame=frame, channel=chan).shape == (N_bin,)
        assert h.get_file_list(()) == (fname,)
