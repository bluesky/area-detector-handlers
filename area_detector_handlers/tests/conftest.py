import numpy as np
import pytest
import h5py
from tempfile import TemporaryDirectory
from pathlib import Path


@pytest.fixture(scope="module")
def xs3file(request):
    N_chans = 4
    N_points = 27
    N_bin = 4096
    N_roi = 32

    f_dir = TemporaryDirectory()

    out_name = Path(f_dir.name) / "xs3.h5"

    with h5py.File(out_name, "w") as fout:
        data = np.random.rand(N_points, N_chans, N_bin) * 1000
        fout["/entry/instrument/detector/data"] = data
        bin_per_roi = N_bin // N_roi
        base = "/entry/instrument/detector/NDAttributes"
        for chan in range(1, N_chans + 1):
            for roi in range(1, N_roi + 1):
                HLM = roi * bin_per_roi
                LLM = (roi - 1) * bin_per_roi
                fout[f"{base}/CHAN{chan}ROI{roi}"] = np.sum(
                    data[:, chan - 1, LLM:HLM], axis=1
                )
                fout[f"{base}/CHAN{chan}ROI{roi}HLM"] = np.ones(N_points) * HLM
                fout[f"{base}/CHAN{chan}ROI{roi}LLM"] = np.ones(N_points) * LLM
        fout[f"{base}/TIMESTAMP"] = np.arange(N_points)
        fout[f"{base}/ImageCounter"] = np.arange(N_points)

    def finalize():
        f_dir.cleanup()

    request.addfinalizer(finalize)
    return (out_name, {}), (N_points, N_chans, N_bin, N_roi)
