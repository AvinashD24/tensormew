import argparse
import unittest
from unittest import mock

from tools import cloud_train_deep as c


def make_args(**kw):
    base = dict(
        host="1.2.3.4", port="2222", key="k", updates=10, rollout=128, max_steps=1024, envs=16, batch_size=256,
        lr=3e-4, state="route29_entry", states="", curriculum=True, recurrent=True, save="logs/deep_policy.pt",
        resume=False, run_id="", config="", sync_every=1.0, watch=False,
    )
    base.update(kw)
    return argparse.Namespace(**base)


GOOD = "GPU=NVIDIA GeForce RTX 4090, 24564 MiB\nCORES=8\nDISK_GB=120\nDEP_torch=ok\nDEP_numpy=ok\nDEP_pyboy=missing\nDEP_PIL=ok\n"


class ParsingTests(unittest.TestCase):
    def test_parse_preflight(self):
        info = c.parse_preflight(GOOD)
        self.assertIn("4090", info["gpu"])
        self.assertEqual((info["cores"], info["disk_gb"]), (8, 120))
        self.assertEqual(info["missing"], ["pyboy"])
        self.assertEqual(c.preflight_problems(info), [])

    def test_no_gpu_and_low_disk(self):
        info = c.parse_preflight("GPU=\nCORES=4\nDISK_GB=2\nDEP_torch=missing\nDEP_numpy=ok\nDEP_pyboy=ok\nDEP_PIL=ok\n")
        probs = " ".join(c.preflight_problems(info))
        self.assertIn("GPU", probs)
        self.assertIn("2 GB", probs)
        self.assertIn("torch", probs)

    def test_cap_envs(self):
        self.assertEqual(c.cap_envs(16, 8)[0], 8)
        self.assertIsNotNone(c.cap_envs(16, 8)[1])
        self.assertEqual(c.cap_envs(8, 16), (8, None))
        self.assertEqual(c.cap_envs(16, None), (16, None))

    def test_parse_status(self):
        self.assertEqual(c.parse_status("running\n"), ("running", None))
        self.assertEqual(c.parse_status("done:0\n"), ("done", 0))
        self.assertEqual(c.parse_status("done:1"), ("done", 1))
        self.assertEqual(c.parse_status("dead"), ("dead", None))
        self.assertEqual(c.parse_status(""), ("unknown", None))


class BuilderTests(unittest.TestCase):
    def test_install_cmd(self):
        cmd = c.build_install_cmd()
        self.assertIn("--break-system-packages", cmd)
        self.assertIn("pyboy==2.8.1", cmd)

    def test_train_cmd_keeps_existing_flags(self):
        cmd = c.build_remote_train_cmd(make_args(states="a,b", resume=True, run_id="r1"), envs=8)
        for frag in ("tools.train_deep_ppo", "--updates 10", "--envs 8", "--states a,b", "--resume", "--run-id r1",
                     "--curriculum", "--recurrent", "--save logs/deep_policy.pt"):
            self.assertIn(frag, cmd)
        self.assertNotIn("--config", cmd)
        self.assertTrue(cmd.startswith("cd /workspace/tensormew && PYTHONUNBUFFERED=1 python3"))

    def test_train_cmd_config_and_negations(self):
        cmd = c.build_remote_train_cmd(make_args(curriculum=False, recurrent=False), 4, "blueprint_v1")
        self.assertIn("--config blueprint_v1", cmd)
        self.assertIn("--no-curriculum", cmd)
        self.assertIn("--no-recurrent", cmd)

    def test_config_forwards_only_explicit_flags(self):
        args = make_args(updates=7, lr=1e-3)
        args.explicit = {"updates", "config"}
        cmd = c.build_remote_train_cmd(args, 16, "blueprint_v1")
        self.assertIn("--updates 7", cmd)
        self.assertIn("--config blueprint_v1", cmd)
        for frag in ("--lr", "--rollout", "--envs", "--max-steps", "--batch-size", "--state ", "--save",
                     "--curriculum", "--recurrent"):
            self.assertNotIn(frag, cmd)
        # a pod-capped env count must still be forwarded
        self.assertIn("--envs 4", c.build_remote_train_cmd(args, 4, "blueprint_v1"))
        # without --config every flag is forwarded even if `explicit` is set
        self.assertIn("--lr", c.build_remote_train_cmd(args, 16, None))

    def test_explicit_dests(self):
        import argparse as ap

        p = ap.ArgumentParser()
        p.add_argument("--updates", type=int, default=1)
        p.add_argument("--curriculum", action="store_true", default=True)
        p.add_argument("--no-curriculum", action="store_false", dest="curriculum")
        p.add_argument("--lr", type=float, default=1.0)
        self.assertEqual(c.explicit_dests(p, ["--no-curriculum", "--lr=0.1"]), {"curriculum", "lr"})
        self.assertEqual(c.explicit_dests(p, []), set())

    def test_runner_script(self):
        script = c.build_runner_script("echo hi")
        self.assertIn("train.log", script)
        self.assertIn("/done", script)
        self.assertIn("echo $rc", script)

    def test_launch_and_status(self):
        launch = c.build_launch_cmd()
        self.assertIn("tmux", launch)
        self.assertIn("setsid nohup", launch)
        self.assertIn("kill -0", c.build_status_cmd())

    def test_ssh_and_pull(self):
        self.assertEqual(c.build_ssh_cmd("k", "22", "h", "ls")[-2:], ["root@h", "ls"])
        pull = c.build_pull_cmd("ssh -p 1", "h", "logs/live/", "logs/live/")
        self.assertEqual(pull[0], "rsync")
        self.assertEqual(pull[-2], "root@h:/workspace/tensormew/logs/live/")
        self.assertIn("/logs/", c.build_sync_up_cmd("ssh", "h"))


class PreflightFlowTests(unittest.TestCase):
    def test_installs_pyboy_and_caps_envs(self):
        calls = []

        def fake(key, port, host, remote, timeout=60, stdin=None):
            calls.append(remote)
            if remote == c.PREFLIGHT_SCRIPT:
                return 0, GOOD
            if remote.startswith("pip install"):
                return 0, ""
            return 0, "ok\n"

        with mock.patch.object(c, "remote_capture", side_effect=fake):
            ok, envs = c.run_preflight(make_args(envs=16))
        self.assertTrue(ok)
        self.assertEqual(envs, 8)
        self.assertTrue(any(x.startswith("pip install --break-system-packages") for x in calls))

    def test_unreachable(self):
        with mock.patch.object(c, "remote_capture", return_value=(255, "")):
            ok, _ = c.run_preflight(make_args())
        self.assertFalse(ok)

    def test_install_failure(self):
        def fake(key, port, host, remote, timeout=60, stdin=None):
            return (0, GOOD) if remote == c.PREFLIGHT_SCRIPT else (1, "")

        with mock.patch.object(c, "remote_capture", side_effect=fake):
            ok, _ = c.run_preflight(make_args())
        self.assertFalse(ok)


class PollTests(unittest.TestCase):
    def run_poll(self, outputs):
        it = iter(outputs)

        def fake(key, port, host, remote, timeout=60, stdin=None):
            if remote == c.build_status_cmd():
                return next(it)
            return 0, "log line"

        with mock.patch.object(c, "remote_capture", side_effect=fake), mock.patch.object(c.time, "sleep"):
            return c.poll_detached(make_args(), "ssh", max_failures=3)

    def test_done_ok_and_failed(self):
        self.assertEqual(self.run_poll([(0, "running"), (0, "done:0")]), c.EXIT_OK)
        self.assertEqual(self.run_poll([(0, "done:2")]), c.EXIT_TRAIN)

    def test_dead_and_unreachable(self):
        self.assertEqual(self.run_poll([(0, "dead")]), c.EXIT_LOST)
        self.assertEqual(self.run_poll([(255, "")] * 3), c.EXIT_LOST)

    def test_transient_failure_recovers(self):
        self.assertEqual(self.run_poll([(255, ""), (0, "done:0")]), c.EXIT_OK)


class MainTests(unittest.TestCase):
    def test_missing_address(self):
        with mock.patch.object(c, "DEFAULT_POD_HOST", ""), self.assertRaises(SystemExit):
            c.main(["--host", "", "--port", ""])

    def test_attach_detach_exclusive(self):
        with self.assertRaises(SystemExit):
            c.main(["--host", "h", "--port", "1", "--attach", "--detach"])

    def test_bad_config_exit_code(self):
        with self.assertRaises(SystemExit) as cm:
            c.main(["--host", "h", "--port", "1", "--config", "no_such_preset"])
        self.assertEqual(cm.exception.code, c.EXIT_PREFLIGHT)

    def test_attach_without_job(self):
        with mock.patch.object(c, "remote_capture", return_value=(0, "none")):
            self.assertEqual(c.main(["--host", "h", "--port", "1", "--attach"]), c.EXIT_FAIL)

    def test_abbreviated_flags_are_rejected(self):
        with self.assertRaises(SystemExit):
            c.main(["--host", "h", "--port", "1", "--res"])  # would silently become --resume / --resume-less

    def test_resume_without_any_checkpoint_refuses_before_training(self):
        calls = []
        with mock.patch.object(c, "run_cmd", side_effect=lambda cmd: calls.append(cmd) or 0), \
                mock.patch.object(c, "remote_capture", return_value=(1, "")):
            rc = c.main(["--host", "h", "--port", "1", "--resume", "--save", "logs/nope_does_not_exist.pt",
                         "--skip-preflight"])
        self.assertEqual(rc, c.EXIT_PREFLIGHT)
        self.assertFalse(any("tools.train_deep_ppo" in " ".join(x) for x in calls))


class ResumeUploadTests(unittest.TestCase):
    def test_checkpoint_files_include_every_sidecar(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            for name in ("run.pt", "run.optim.pt", "run.scaler.pt", "run2.pt", "other.pt"):
                (Path(d) / name).write_bytes(b"x")
            got = [f.name for f in c.checkpoint_files(str(Path(d) / "run.pt"))]
            self.assertEqual(got, ["run.optim.pt", "run.pt", "run.scaler.pt"])

    def test_resume_uploads_checkpoint_and_sidecars_then_forces_save(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            save = str(Path(d) / "run.pt")
            for name in ("run.pt", "run.optim.pt"):
                (Path(d) / name).write_bytes(b"x")
            cmds = []
            with mock.patch.object(c, "run_cmd", side_effect=lambda cmd: cmds.append(cmd) or 0), \
                    mock.patch.object(c, "remote_capture", return_value=(0, "")):
                self.assertEqual(c.upload_resume_checkpoint(make_args(save=save, resume=True), "ssh"), c.EXIT_OK)
            up = " ".join(cmds[0])
            self.assertIn("run.optim.pt", up)
            self.assertIn("run.pt", up)
            self.assertIn("root@1.2.3.4:", up)

    def test_resume_upload_failure_blocks_training(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            save = Path(d) / "run.pt"
            save.write_bytes(b"x")
            with mock.patch.object(c, "run_cmd", return_value=23), mock.patch.object(c, "remote_capture", return_value=(0, "")):
                self.assertEqual(c.upload_resume_checkpoint(make_args(save=str(save), resume=True), "ssh"), c.EXIT_FAIL)

    def test_pull_keeps_previous_weights_on_resume_and_pulls_sidecars(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as d:
            save = Path(d) / "run.pt"
            save.write_bytes(b"old")
            cmds = []
            with mock.patch.object(c, "run_cmd", side_effect=lambda cmd: cmds.append(cmd) or 0):
                self.assertEqual(c.pull_checkpoint(make_args(save=str(save), resume=True), "ssh"), c.EXIT_OK)
            self.assertEqual((Path(d) / "run.pt.pre-resume").read_bytes(), b"old")
            self.assertEqual(len(cmds), 2)
            self.assertTrue(cmds[1][-2].endswith("/run.*"))  # sidecar glob on the pod


if __name__ == "__main__":
    unittest.main()
