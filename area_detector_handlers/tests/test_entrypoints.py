import pkg_resources


def test_entrypoint_consisetncy():
    for ep in pkg_resources.iter_entry_points("databroker.handlers"):
        handler = ep.load()
        assert ep.name in handler.specs
