import os
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory, gettempprefix

import entrypoints
import h5py
import numpy as np
import pytest
import tifffile

from area_detector_handlers.eiger import EigerHandler


def select_handler(spec):
    handlers = [
        ep.load()
        for ep in entrypoints.get_group_all("databroker.handlers")
        if ep.name == spec
    ]
    assert len(handlers)
    return pytest.mark.parametrize("handler", handlers)


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


@pytest.fixture(scope="module")
def eigerfile(request):
    N_chans = 4
    N_points = 27
    N_bin = 4096
    N_roi = 32

    seq_id = np.random.randint(10)
    pre_fix = gettempprefix()
    out_name = f'{pre_fix}_{seq_id}_master.h5'

    with h5py.File("sample_data_000001.h5", "w") as dfile:
        data = np.random.rand(N_points, N_chans, N_bin) * 1000
        dfile["data_000001"] = data

    with h5py.File(out_name, "w") as fout:
        for e in EigerHandler.EIGER_MD_LAYOUT.values():
            fout[e] = np.random.randint(1, 10)
        fout["entry/data/data_000001"] = h5py.ExternalLink("sample_data_000001.h5", "data_000001")

    def finalize():
        os.remove(out_name)
        os.remove("sample_data_000001.h5")

    request.addfinalizer(finalize)
    kwargs = {'images_per_file': 1, 'frame_per_point': 1}
    return (out_name, kwargs), (N_points, N_chans, N_bin, N_roi)


@pytest.fixture(scope="module", params=[1, 5])
def hdf5_files(request):
    N_rows = 13
    N_cols = 27
    N_points = 7
    fpp = request.param
    f_dir = TemporaryDirectory()
    fname = f"adh_{fpp}_test.h5"
    full_path = str(Path(f_dir.name, fname))

    data = np.concatenate([
        np.ones((fpp, N_rows, N_cols)) * pt for pt in range(N_points)])
    timestamps = np.concatenate([
        np.linspace(n * fpp, (n + 1) * fpp - 1, fpp) for n in range(N_points)])
    timestamps_sec = timestamps.astype(np.int64)
    # Add 1e-9 to the timestamps to test nanosecond precision
    timestamps_ns = ((timestamps - timestamps_sec) * 1e9) + 1
    with h5py.File(full_path, "w") as file:
        file.create_dataset('entry/data/data', data=data)
        file.create_dataset('entry/instrument/NDAttributes/NDArrayTimeStamp', data=timestamps)
        file.create_dataset('entry/instrument/NDAttributes/NDArrayEpicsTSSec', data=timestamps_sec)
        file.create_dataset('entry/instrument/NDAttributes/NDArrayEpicsTSnSec', data=timestamps_ns)

    def finalize():
        f_dir.cleanup()

    request.addfinalizer(finalize)

    return (
        (full_path, dict(frame_per_point=fpp)),
        (N_rows, N_cols, N_points, fpp),
    )


@pytest.fixture(scope="module", params=[1, 5])
def tiff_files(request):
    N_rows = 13
    N_cols = 27
    N_points = 7
    fpp = request.param
    f_dir = TemporaryDirectory()
    template = "%s/%s_%05d.tiff"
    fname = f"adh_{fpp}_test"
    file_index = count()

    for pt in range(N_points):
        for i in range(fpp):
            write_fname = template % (f_dir.name, fname, next(file_index))
            tifffile.imwrite(write_fname,
                             np.ones((N_rows, N_cols)) * pt,
                             # Specifies use of the EPICS metadata
                             software="EPICS areaDetector",
                             # Adds the EPICS timestamp to the TIFF file as a tag
                             extratags=[
                                 # The detector timestamp
                                 (65000, tifffile.DATATYPE.FLOAT, 1, [pt * fpp + i], True),
                                 # The EPICS driver timestamp in seconds
                                 (65002, tifffile.DATATYPE.LONG, 1, [pt * fpp + i], True),
                                 # The EPICS driver timestamp in nanoseconds
                                 (65003, tifffile.DATATYPE.LONG, 1, [i], True)])

    def finalize():
        f_dir.cleanup()

    request.addfinalizer(finalize)

    return (
        (f_dir.name, dict(template=template, frame_per_point=fpp, filename=fname)),
        (N_rows, N_cols, N_points, fpp),
    )
