from area_detector_handlers.tests.conftest import select_handler
import pytest


@select_handler("XSP3_FLY")
def test_bulk(xs3file, handler):
    (fname, kwargs), (N_points, N_chans, N_bin, N_roi) = xs3file

    with handler(fname, **kwargs) as h:
        assert h().shape == (N_points, N_chans, N_bin)
        assert h(target="data").shape == (N_points, N_chans, N_bin)
        for chan in range(1, N_chans + 1):
            for roi in range(1, N_roi + 1):
                assert h(target=f"CHAN{chan}ROI{roi}").shape == (N_points,)
                assert h(target=f"CHAN{chan}ROI{roi}HLM").shape == (N_points,)
                assert h(target=f"CHAN{chan}ROI{roi}LLM").shape == (N_points,)

        assert h.get_file_list(()) == (fname,)


@select_handler("XSP3")
def test_pre_pixel(xs3file, handler):
    (fname, kwargs), (N_points, N_chans, N_bin, N_roi) = xs3file

    with handler(fname, **kwargs) as h:
        for frame in range(0, N_points):
            for chan in range(1, N_chans + 1):
                assert h(frame=frame, channel=chan).shape == (N_bin,)
        assert h.get_file_list(()) == (fname,)
