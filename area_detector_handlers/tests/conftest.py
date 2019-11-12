import numpy as np
import pytest
import h5py
import tifffile
from itertools import count
from tempfile import TemporaryDirectory
from pathlib import Path
import entrypoints


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
    with h5py.File(full_path) as file:
        file.create_dataset('entry/data/data', data=data)

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

    print(f_dir.name)
    for pt in range(N_points):
        for _ in range(fpp):
            write_fname = template % (f_dir.name, fname, next(file_index))
            tifffile.imwrite(write_fname, np.ones((N_rows, N_cols)) * pt)
            print(write_fname)

    def finalize():
        f_dir.cleanup()

    print("*")
    for f in Path(f_dir.name).rglob("*.tiff"):
        print(f)
    print("*")
    request.addfinalizer(finalize)

    return (
        (f_dir.name, dict(template=template, frame_per_point=fpp, filename=fname)),
        (N_rows, N_cols, N_points, fpp),
    )
