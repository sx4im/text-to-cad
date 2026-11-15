"""Ctrl+C ends the stdio transport quietly: an interrupt that lands while serving, or while
the session is being ended in cleanup, leaves with no traceback and exit 0 -- the same
quiet end as the host closing its input, not a crash in the host's log."""

from __future__ import annotations

import io
import threading
import unittest
from unittest import mock

from cad_mcp import shim


class FakeHolder:
    """A hold already taken: ``serve`` never waits on the engine."""

    def __init__(self, *args, **kwargs):
        self.tried = threading.Event()
        self.tried.set()

    def start(self):
        pass


class FakeProxy:
    """A session whose end can be interrupted, like the real one in ``proxy.close``."""

    interrupt_close = False

    def __init__(self, *args, **kwargs):
        self.closed = False

    def handle(self, *args, **kwargs):
        return {}

    def notify(self, *args, **kwargs):
        pass

    def close(self):
        self.closed = True
        if self.interrupt_close:
            raise KeyboardInterrupt


class FakeConnection:
    """A stdio reader that ends, or is interrupted, on the test's word."""

    interrupt_serve = False

    def __init__(self, *args, **kwargs):
        pass

    def serve(self):
        if self.interrupt_serve:
            raise KeyboardInterrupt


class ShimServeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.proxy = FakeProxy()
        patches = [
            mock.patch.object(shim, "claim_stdin", return_value=io.BytesIO(b"")),
            mock.patch.object(shim, "claim_stdout", return_value=io.BytesIO()),
            mock.patch.object(shim, "launcher", return_value={"host": "test"}),
            mock.patch.object(shim, "Holder", FakeHolder),
            mock.patch.object(shim, "Proxy", return_value=self.proxy),
            mock.patch.object(shim, "Connection", FakeConnection),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def test_an_interrupt_while_serving_ends_the_session_quietly(self) -> None:
        FakeConnection.interrupt_serve = True
        self.addCleanup(setattr, FakeConnection, "interrupt_serve", False)
        self.assertEqual(shim.serve(), 0)
        self.assertTrue(self.proxy.closed, "the session is still ended through proxy.close()")

    def test_an_interrupt_during_cleanup_ends_quietly(self) -> None:
        FakeProxy.interrupt_close = True
        self.addCleanup(setattr, FakeProxy, "interrupt_close", False)
        self.assertEqual(shim.serve(), 0)

    def test_the_input_ending_still_ends_the_session_quietly(self) -> None:
        self.assertEqual(shim.serve(), 0)
        self.assertTrue(self.proxy.closed)


if __name__ == "__main__":
    unittest.main()
