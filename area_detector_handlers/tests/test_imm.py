import os
import numpy as np
import pytest
from area_detector_handlers.tests.conftest import select_handler


@pytest.mark.parametrize('imm_file,frames_per_point,frame_shape',
                         [('B020_bluesky_demo_00001-00021.imm',
                           21, (516, 1556))])
@select_handler("IMM")
def test_imm(imm_file, frames_per_point, frame_shape, handler):
    import area_detector_handlers.tests as tests
    root = os.path.join(os.path.dirname(tests.__file__), 'data')
    path = os.path.join(root, imm_file)
    kwargs = {'frames_per_point': frames_per_point}
    expected_shape = (frames_per_point, *frame_shape)
    with handler(path, **kwargs) as h:
        d = h(index=0)
        assert d.shape == expected_shape
