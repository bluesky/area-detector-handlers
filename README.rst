======================
Area Detector Handlers
======================

.. image:: https://img.shields.io/travis/danielballan/area-detector-handlers.svg
        :target: https://travis-ci.org/danielballan/area-detector-handlers

.. image:: https://img.shields.io/pypi/v/area-detector-handlers.svg
        :target: https://pypi.python.org/pypi/area-detector-handlers


DataBroker "handlers" for Area Detector

This are classes that read data that was written by Area Detector. They use
information encoded in Resource and Datum documents to locate files and slices
within files.

Installing this package makes it automatically discoverable by databroker. It
registers several ``'databroker.handlers'`` entry points.
