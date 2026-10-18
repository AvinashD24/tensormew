"""Terminal Progress Monitor for Reinforcement Learning Training.

Provides clear, visual, aligned terminal telemetry for PPO training runs
with visual entropy gauge bars, rolling reward trends, PPO stability metrics,
and automatic stagnation / policy collapse detection.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any


class TerminalMonitor:
    """Formats and prints live RL metrics to terminal and appends to JSONL."""

    def __init__(
        self,
        *,
        total_updates: int | None = None,
        jsonl_path: str | Path | None = "logs/training_metrics.jsonl",
        header_interval: int = 15,
        max_entropy: float = math.log(5),  # 5 discrete actions: ln(5) ~ 1.609
    ):
        self.total_updates = total_updates
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.header_interval = header_interval
        self.max_entropy = max_entropy

        self.reward_window: deque[float] = deque(maxlen=5)
        self.last_rolling_avg: float | None = None
        self.last_maps_discovered: int = 0
        self.updates_since_map_change: int = 0
        self.update_count: int = 0

        # Colors enabled if terminal supports it
        self.use_color = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    def _color(self, text: str, code: str) -> str:
        if not self.use_color:
            return text
        return f"\033[{code}m{text}\033[0m"

    def format_header(self) -> str:
        """Returns aligned table header."""
        border = "=" * 120
        header = (
            f" {'Upd':>4s} | "
            f"{'Reward (Roll5)':^17s} | "
            f"{'Entropy H(π)':^17s} | "
            f"{'Val Loss':>9s} | "
            f"{'Appx KL':>7s} | "
            f"{'Clip%':>6s} | "
            f"{'LR':>7s} | "
            f"{'Maps':>4s} | "
            f"{'Time':>5s} | "
            f"{'Status / Behavior':^23s}"
        )
        sub_border = "-" * 120
        return f"{border}\n{header}\n{sub_border}"

    def assess_status(
        self,
        *,
        mean_reward: float,
        entropy: float,
        approx_kl: float,
        maps_discovered: int,
    ) -> tuple[str, str]:
        """Diagnose training state for stagnation, thrashing, or advancement."""
        # 1. Critical Policy Collapse: Entropy collapsed to near 0
        if entropy < 0.35:
            return "COLLAPSED", self._color("🔴 STAGNANT (COLLAPSED)", "1;31")

        # 2. Destructive Thrashing: KL is too large, policy overwriting itself
        if approx_kl > 0.08:
            return "THRASHING", self._color("⚠️  THRASHING (HIGH KL)", "1;33")

        # 3. New Map or Major Progress Breakthrough
        if maps_discovered > self.last_maps_discovered or mean_reward >= 100.0:
            return "ADVANCING", self._color("🟢 ADVANCING", "1;32")

        # 4. Looping / Stuck in local negative reward cycle
        if self.updates_since_map_change >= 10 and mean_reward < 0.0:
            return "LOOPING", self._color("🟡 LOOPING (STUCK)", "33")

        # 5. Normal healthy exploration
        if entropy >= 0.70:
            return "EXPLORING", self._color("🔵 EXPLORING", "36")

        return "LEARNING", self._color("🟢 LEARNING", "32")

    def format_entropy_bar(self, entropy: float, bar_width: int = 8) -> str:
        """Renders an ASCII block gauge representing policy action entropy."""
        frac = min(1.0, max(0.0, entropy / self.max_entropy))
        filled = int(round(frac * bar_width))
        bar = "█" * filled + "░" * (bar_width - filled)

        # Color the bar based on health
        if entropy < 0.40:
            colored_bar = self._color(f"[{bar}]", "31")  # Red: Collapse danger
        elif entropy < 0.70:
            colored_bar = self._color(f"[{bar}]", "33")  # Yellow: Low exploration
        else:
            colored_bar = self._color(f"[{bar}]", "36")  # Cyan: Healthy exploration

        return f" {colored_bar} {entropy:4.2f} "

    def format_reward_trend(self, mean_reward: float) -> str:
        """Formats reward with rolling average and direction trend arrow."""
        self.reward_window.append(mean_reward)
        roll_avg = sum(self.reward_window) / len(self.reward_window)

        if self.last_rolling_avg is None:
            trend = " "
        elif roll_avg > self.last_rolling_avg + 1.0:
            trend = self._color("▲", "1;32")  # Improving
        elif roll_avg < self.last_rolling_avg - 1.0:
            trend = self._color("▼", "1;31")  # Declining
        else:
            trend = self._color("▬", "90")    # Flat

        self.last_rolling_avg = roll_avg

        # Color reward number
        rew_str = f"{mean_reward:+7.2f}"
        if mean_reward > 0:
            rew_colored = self._color(rew_str, "32")
        elif mean_reward < -5:
            rew_colored = self._color(rew_str, "33")
        else:
            rew_colored = rew_str

        return f"{rew_colored} ({roll_avg:+6.1f}){trend}"

    def log_update(
        self,
        *,
        update: int,
        mean_reward: float,
        entropy: float,
        value_loss: float,
        policy_loss: float = 0.0,
        approx_kl: float = 0.0,
        clip_frac: float = 0.0,
        lr: float = 3e-4,
        maps_discovered: int = 0,
        badges: int = 0,
        duration_sec: float = 0.0,
        samples: int = 1024,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Formats, prints, and logs one training update."""
        # Track map discovery updates
        if maps_discovered > self.last_maps_discovered:
            self.last_maps_discovered = maps_discovered
            self.updates_since_map_change = 0
        else:
            self.updates_since_map_change += 1

        # Check if header should be printed
        lines: list[str] = []
        if self.update_count % self.header_interval == 0:
            lines.append(self.format_header())
        self.update_count += 1

        # Calculate status
        status_key, status_display = self.assess_status(
            mean_reward=mean_reward,
            entropy=entropy,
            approx_kl=approx_kl,
            maps_discovered=maps_discovered,
        )

        # Formatted fields
        upd_str = f"{update:4d}"
        rew_display = self.format_reward_trend(mean_reward)
        ent_display = self.format_entropy_bar(entropy)
        vloss_str = f"{value_loss:9.2f}"
        kl_str = f"{approx_kl:7.4f}"
        clip_str = f"{clip_frac * 100:5.1f}%"
        lr_compact = f"{lr:.1e}".replace("e-0", "e-").replace("e+0", "e+")
        lr_str = f"{lr_compact:>7s}"
        maps_str = f"{maps_discovered:4d}"
        time_str = f"{duration_sec:4.1f}s"

        row = (
            f" {upd_str} | "
            f"{rew_display} | "
            f"{ent_display} | "
            f"{vloss_str} | "
            f"{kl_str} | "
            f"{clip_str} | "
            f"{lr_str} | "
            f"{maps_str} | "
            f"{time_str} | "
            f"{status_display}"
        )
        lines.append(row)
        output = "\n".join(lines)
        print(output, flush=True)

        # Append to JSONL log
        if self.jsonl_path:
            try:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                record = {
                    "update": update,
                    "mean_reward": round(mean_reward, 3),
                    "rolling_reward": round(float(self.last_rolling_avg or mean_reward), 3),
                    "entropy": round(entropy, 4),
                    "value_loss": round(value_loss, 4),
                    "policy_loss": round(policy_loss, 4),
                    "approx_kl": round(approx_kl, 5),
                    "clip_frac": round(clip_frac, 4),
                    "lr": lr,
                    "maps_discovered": maps_discovered,
                    "badges": badges,
                    "duration_sec": round(duration_sec, 2),
                    "samples": samples,
                    "status": status_key,
                    "timestamp": time.time(),
                }
                if extra:
                    record.update(extra)
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                pass

        return output
