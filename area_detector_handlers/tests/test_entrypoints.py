import entrypoints


def test_entrypoint_consisetncy():
    j = 0
    for ep in entrypoints.get_group_all("databroker.handlers"):
        j += 1
        handler = ep.load()
        assert ep.name in handler.specs
    assert j != 0
