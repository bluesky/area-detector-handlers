from os import path
from setuptools import setup, find_packages
import sys
import versioneer


# NOTE: This file must remain Python 2 compatible for the foreseeable future,
# to ensure that we error out properly for people with outdated setuptools
# and/or pip.
min_version = (3, 6)
if sys.version_info < min_version:
    error = """
area-detector-handlers does not support Python {0}.{1}.
Python {2}.{3} and above is required. Check your Python version like so:

python3 --version

This may be due to an out-of-date pip. Make sure you have pip >= 9.0.1.
Upgrade pip like so:

pip install --upgrade pip
""".format(*(sys.version_info[:2] + min_version))
    sys.exit(error)

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as readme_file:
    readme = readme_file.read()

with open(path.join(here, 'requirements.txt')) as requirements_file:
    # Parse requirements.txt, ignoring any commented-out lines.
    requirements = [line for line in requirements_file.read().splitlines()
                    if not line.startswith('#')]


setup(
    name='area-detector-handlers',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="DataBroker 'handlers' for Area Detector",
    long_description=readme,
    author="Brookhaven National Lab",
    author_email="dallan@bnl.gov",
    url="https://github.com/danielballan/area-detector-handlers",
    python_requires=">={}".format(".".join(str(n) for n in min_version)),
    packages=find_packages(exclude=["docs", "tests"]),
    entry_points={
        "databroker.handlers": [
            "AD_SPE = area_detector_handlers.handlers:AreaDetectorSPEHandler",
            "AD_TIFF = area_detector_handlers.handlers:AreaDetectorTiffHandler",
            "AD_TIFF_EPICS_TS = area_detector_handlers.handlers:AreaDetectorTiffEpicsTimestampHandler",
            "AD_EIGER = area_detector_handlers.eiger:EigerHandler",
            "AD_EIGER2 = area_detector_handlers.eiger:EigerHandler",
            "AD_EIGER_SLICE = area_detector_handlers.eiger:EigerHandler",
            "AD_HDF5 = area_detector_handlers.handlers:AreaDetectorHDF5Handler",
            "AD_HDF5_SWMR = area_detector_handlers.handlers:AreaDetectorHDF5SWMRHandler",
            "AD_HDF5_TS = area_detector_handlers.handlers:AreaDetectorHDF5TimestampHandler",
            "AD_HDF5_NDARRAY_TS = area_detector_handlers.handlers:AreaDetectorHDF5NDArrayTimestampHandler",
            "AD_HDF5_SWMR_TS = area_detector_handlers.handlers:AreaDetectorHDF5SWMRTimestampHandler",
            "AD_HDF5_SINGLE = area_detector_handlers.handlers:AreaDetectorHDF5SingleHandler",
            "SPECS_HDF5_SINGLE_DATAFRAME = area_detector_handlers.handlers:SpecsHDF5SingleHandlerDataFrame",
            "XSP3 = area_detector_handlers.handlers:Xspress3HDF5Handler",
            "TPX_HDF5 = area_detector_handlers.handlers:TimepixHDF5Handler",
            "AD_CBF = area_detector_handlers.handlers:PilatusCBFHandler",
            "XSP3_FLY = area_detector_handlers.handlers:BulkXSPRESS",
            "XPS3_FLY = area_detector_handlers.handlers:BulkXSPRESS",
            "DEXELA_FLY_V1 = area_detector_handlers.handlers:BulkXSPRESS",
            "MERLIN_FLY_STREAM_V1 = area_detector_handlers.handlers:BulkXSPRESS",
            "MERLIN_FLY = area_detector_handlers.handlers:BulkXSPRESS",
            "IMM = area_detector_handlers.handlers:IMMHandler",
        ]
    },
    include_package_data=True,
    package_data={
        'area_detector_handlers': [
            # When adding files here, remember to update MANIFEST.in as well,
            # or else they will not be included in the distribution on PyPI!
            # 'path/to/data_file',
            'area_detector_handlers/tests/data/*',
        ]
    },
    install_requires=requirements,
    license="BSD (3-clause)",
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)
