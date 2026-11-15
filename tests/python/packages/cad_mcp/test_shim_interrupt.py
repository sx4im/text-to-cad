"""Ctrl+C ends the stdio transport quietly (``shim.main``): the session is ended through ``proxy.close``
and the process exits zero, with no traceback -- the same quiet end as the host closing its input, not a
crash in the host's log. Nothing starts for real: ``serve``'s collaborators are stood in for."""

from __future__ import annotations

import unittest
from unittest import mock

from cad_mcp import shim


class InterruptTest(unittest.TestCase):
    def run_main(self, *, serve_raises=None, close_raises=None):
        """``shim.main`` with ``serve``'s collaborators stood in for: the read loop ends, or raises,
        as told, and the session's end raises, or not, as told."""
        connection = mock.Mock()
        connection.serve.side_effect = serve_raises
        proxy = mock.Mock()
        proxy.close.side_effect = close_raises
        holder = mock.Mock()
        holder.tried.wait.return_value = True
        with mock.patch.object(shim, "claim_stdout", return_value=mock.Mock()), \
                mock.patch.object(shim, "claim_stdin", return_value=mock.Mock()), \
                mock.patch.object(shim, "launcher", return_value=None), \
                mock.patch.object(shim, "Holder", return_value=holder), \
                mock.patch.object(shim, "Proxy", return_value=proxy), \
                mock.patch.object(shim, "Connection", return_value=connection):
            return shim.main([]), proxy

    def test_an_interrupt_inside_the_session_s_end_exits_zero_without_a_traceback(self) -> None:
        # The issue's race: the host closes the transport, the read loop ends, and Ctrl+C lands inside
        # the finally block's proxy.close(). The interrupt escapes serve() -- and main() ends the
        # transport quietly instead of printing it.
        code, _proxy = self.run_main(close_raises=KeyboardInterrupt)
        self.assertEqual(code, 0)

    def test_an_interrupt_during_the_read_loop_still_ends_the_session_and_exits_zero(self) -> None:
        # Ctrl+C while the host's input is still open: the session is ended through proxy.close() on
        # the way out, and the process exits zero all the same.
        code, proxy = self.run_main(serve_raises=KeyboardInterrupt)
        self.assertEqual(code, 0)
        proxy.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
