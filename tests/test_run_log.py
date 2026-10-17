import unittest
from pathlib import Path

from src.run_log import RunLog


class RunLogTests(unittest.TestCase):
    def test_records_progress_and_failed_reason(self):
        path = Path("logs/test/run_log.db")
        if path.exists():
            path.unlink()
        log = RunLog(path)
        try:
            run_id = log.start_run(state_name="new_bark_outside", start_map=(24, 4), start_badges=0)
            log.add_step(
                run_id,
                {
                    "frame": 10,
                    "map": [24, 4],
                    "pos": [3, 9],
                    "badges": 0,
                    "actor": "script",
                    "trigger": "",
                    "tool": "run_path",
                    "arguments": {"directions": ["left"]},
                    "reason": "segment script toward map (24, 3)",
                    "ok": True,
                },
            )
            log.add_step(
                run_id,
                {
                    "frame": 40,
                    "map": [24, 4],
                    "pos": [3, 9],
                    "badges": 0,
                    "actor": "llm",
                    "trigger": "DIALOGUE: script_mode=1",
                    "tool": "mash_text",
                    "arguments": {},
                    "reason": "textbox still open",
                    "ok": False,
                    "est_usd": 0.001,
                    "input_tokens": 20,
                    "output_tokens": 8,
                },
            )
            log.finish_run(
                run_id,
                outcome="stuck",
                end_map=(24, 4),
                end_badges=0,
                steps=2,
                cut_cleared=False,
                error="",
            )
            stats = log.progress()
            self.assertEqual(stats["runs"], 1)
            self.assertEqual(stats["stuck"], 1)
            self.assertEqual(stats["max_badges"], 0)
            failed = log.failed_steps()
            self.assertEqual(len(failed), 1)
            self.assertEqual(failed[0]["tool"], "mash_text")
            self.assertIn("textbox", failed[0]["reason"])
            steps = log.run_steps(run_id)
            self.assertEqual(len(steps), 2)
            self.assertEqual(steps[0]["actor"], "script")
        finally:
            log.close()


if __name__ == "__main__":
    unittest.main()
