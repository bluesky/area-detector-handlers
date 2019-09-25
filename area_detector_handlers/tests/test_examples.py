from area_detector_handlers.handlers import HandlerBase


def test_context():
    class Test(HandlerBase):
        def close(self):
            self.called = True

    h = Test()

    with h:
        ...

    assert h.called
