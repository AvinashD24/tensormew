import json
from pathlib import Path

from src.terminal_monitor import TerminalMonitor


def test_terminal_monitor_format_header():
    monitor = TerminalMonitor()
    header = monitor.format_header()
    assert "Upd" in header
    assert "Reward" in header
    assert "Entropy" in header
    assert "Val Loss" in header
    assert "Appx KL" in header
    assert "Clip%" in header
    assert "LR" in header
    assert "Maps" in header
    assert "Status" in header


def test_terminal_monitor_status_assessment():
    monitor = TerminalMonitor()
    # Test collapse detection
    key, display = monitor.assess_status(mean_reward=-5.0, entropy=0.20, approx_kl=0.01, maps_discovered=2)
    assert key == "COLLAPSED"
    assert "COLLAPSED" in display

    # Test thrashing detection
    key, display = monitor.assess_status(mean_reward=10.0, entropy=1.2, approx_kl=0.12, maps_discovered=2)
    assert key == "THRASHING"

    # Test advancing detection
    monitor.last_maps_discovered = 2
    key, display = monitor.assess_status(mean_reward=15.0, entropy=1.1, approx_kl=0.01, maps_discovered=3)
    assert key == "ADVANCING"


def test_terminal_monitor_log_update_writes_jsonl(tmp_path: Path):
    log_file = tmp_path / "metrics.jsonl"
    monitor = TerminalMonitor(jsonl_path=log_file, header_interval=5)

    out = monitor.log_update(
        update=0,
        mean_reward=45.5,
        entropy=1.25,
        value_loss=3.5,
        policy_loss=-0.1,
        approx_kl=0.015,
        clip_frac=0.08,
        lr=3e-4,
        maps_discovered=2,
        duration_sec=10.2,
    )
    assert "45.50" in out
    assert "1.25" in out
    assert log_file.is_file()

    with open(log_file, "r", encoding="utf-8") as f:
        data = json.loads(f.readline())
        assert data["update"] == 0
        assert data["mean_reward"] == 45.5
        assert data["entropy"] == 1.25
        assert data["lr"] == 3e-4
        assert data["value_loss"] == 3.5
        assert data["status"] in ("ADVANCING", "EXPLORING", "LEARNING")
