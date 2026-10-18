import io
import os
import unittest
from contextlib import redirect_stdout

from tools.compare_cost import SAMPLE_TRACE, FakeClient, comparison_line, count_calls, main


class CompareCostTests(unittest.TestCase):
    def test_sample_trace_call_counts(self):
        triggers_only = count_calls(SAMPLE_TRACE, always_on=False, client=FakeClient())
        always_on = count_calls(SAMPLE_TRACE, always_on=True, client=FakeClient())
        self.assertEqual(triggers_only, 2)
        self.assertEqual(always_on, 6)
        self.assertLess(triggers_only, always_on)

    def test_fake_client_records_mash_text(self):
        client = FakeClient()
        response = client.messages.create(model="fake")
        block = response.content[0]
        self.assertEqual(block.type, "tool_use")
        self.assertEqual(block.name, "mash_text")
        self.assertEqual(block.input, {"reason": "test", "max_frames": 30})
        self.assertEqual(len(client.messages.calls), 1)

    def test_comparison_stays_offline(self):
        import socket

        client = FakeClient()
        real_socket = socket.socket

        def refuse(*args, **kwargs):
            raise AssertionError("network socket opened")

        socket.socket = refuse
        try:
            line = comparison_line(list(SAMPLE_TRACE), client)
        finally:
            socket.socket = real_socket
        self.assertEqual(line, "triggers_only=2 always_on=6")
        self.assertEqual(len(client.messages.calls), 8)

    def test_cli_prints_sample_counts(self):
        saved = os.environ.pop("ANTHROPIC_API_KEY", None)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                code = main()
            self.assertEqual(code, 0)
            self.assertEqual(buf.getvalue().strip(), "triggers_only=2 always_on=6")
        finally:
            if saved is not None:
                os.environ["ANTHROPIC_API_KEY"] = saved


if __name__ == "__main__":
    unittest.main()
