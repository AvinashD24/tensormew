import re
import unittest

from tools.train_deep_ppo import build_config, build_parser

SHARED_DEFAULT = "logs/deep_policy.pt"


def _cfg(*argv):
    return build_config(build_parser().parse_args(list(argv)))


class SavePathDefaultTests(unittest.TestCase):
    """A fresh run must not reuse the shared checkpoint name, or it can overwrite older weights."""

    def test_fresh_run_is_named_after_its_run_id(self):
        cfg = _cfg("--run-id", "abc")
        self.assertEqual(cfg.save_path, "logs/abc.pt")
        self.assertEqual(cfg.run_id, "abc")

    def test_without_a_run_id_the_path_is_timestamped_and_matches_the_run_id(self):
        cfg = _cfg()
        self.assertRegex(cfg.save_path, r"^logs/\d{8}-\d{6}\.pt$")
        self.assertEqual(cfg.save_path, f"logs/{cfg.run_id}.pt")

    def test_two_runs_do_not_share_a_path(self):
        self.assertNotEqual(_cfg("--run-id", "a").save_path, _cfg("--run-id", "b").save_path)

    def test_explicit_save_always_wins(self):
        self.assertEqual(_cfg("--save", "logs/mine.pt", "--run-id", "abc").save_path, "logs/mine.pt")
        self.assertEqual(_cfg("--save", SHARED_DEFAULT).save_path, SHARED_DEFAULT)

    def test_resume_keeps_the_shared_default_so_there_is_something_to_load(self):
        self.assertEqual(_cfg("--resume").save_path, SHARED_DEFAULT)

    def test_legacy_preset_gets_a_unique_path_too(self):
        cfg = _cfg("--config", "legacy", "--run-id", "old-style")
        self.assertEqual(cfg.save_path, "logs/old-style.pt")
        self.assertNotRegex(cfg.save_path, re.escape("deep_policy"))


if __name__ == "__main__":
    unittest.main()
