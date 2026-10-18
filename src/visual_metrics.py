"""Visual reward, return, and entropy tracking for live PyBoy runs."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass
class StepRecord:
    step: int
    episode: int
    reward: float
    cumulative_return: float
    entropy: float
    action: int | str = ""
    value: float | None = None
    discovered_tile: bool = False
    discovered_map: bool = False
    badges: int = 0
    map_id: tuple[int, int] | None = None
    timestamp: float = 0.0


@dataclass
class UpdateRecord:
    update: int
    mean_reward: float
    mean_entropy: float
    weight_change: float
    new_maps: int = 0
    timestamp: float = 0.0


class LiveMetricsPlotter:
    """Live interactive Matplotlib monitor for reward, returns, and entropy alongside PyBoy."""

    def __init__(
        self,
        *,
        title: str = "TensorMew: Model Returns & Policy Entropy",
        enabled: bool = True,
        headless: bool = False,
        window_size: int | None = None,
        refresh_fps: float = 30.0,
        save_path: str | Path | None = "logs/reward_entropy_curve.png",
        jsonl_path: str | Path | None = "logs/metrics_history.jsonl",
        n_actions: int = 5,
    ):
        self.enabled = enabled
        self.headless = headless
        self.title = title
        self.window_size = window_size
        self.min_refresh_interval = 1.0 / max(1.0, refresh_fps)
        self.save_path = Path(save_path) if save_path else None
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.n_actions = n_actions
        self.max_entropy = math.log(max(2, n_actions))

        self.step_history: list[StepRecord] = []
        self.update_history: list[UpdateRecord] = []
        self.episode_returns: list[float] = []

        self.current_episode = 0
        self.current_ep_return = 0.0
        self.total_steps = 0
        self.unique_tiles: set[tuple[int, int, int, int]] = set()
        self.unique_maps: set[tuple[int, int]] = set()

        self._last_draw_time = 0.0
        self._fig = None
        self._ax_top = None
        self._ax_bot = None
        self._ax_top_right = None
        self._line_ret = None
        self._line_rolling_ret = None
        self._line_reward = None
        self._line_ent = None
        self._line_rolling_ent = None
        self._hud_text = None
        self._closed = False
        self._initialized_gui = False

        if self.enabled and not self.headless:
            self._init_gui()

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_gui(self) -> None:
        try:
            import matplotlib
            import matplotlib.pyplot as plt

            # Ensure interactive mode is on
            plt.ion()

            # Modern sleek dark theme
            fig, (ax_top, ax_bot) = plt.subplots(
                2, 1, figsize=(9.5, 6.5), facecolor="#0B0F19", sharex=True
            )
            fig.canvas.manager.set_window_title(self.title)

            # Top: Model Returns & Step Rewards
            ax_top.set_facecolor("#111827")
            ax_top.tick_params(colors="#9CA3AF", labelsize=9)
            ax_top.set_ylabel("Episode Return", color="#38BDF8", fontsize=10, fontweight="bold")
            ax_top.grid(True, color="#1F2937", linestyle=":", alpha=0.7)

            ax_top_right = ax_top.twinx()
            ax_top_right.set_facecolor("none")
            ax_top_right.tick_params(colors="#9CA3AF", labelsize=9)
            ax_top_right.set_ylabel("Step Reward", color="#34D399", fontsize=10, fontweight="bold")

            # Lines for Top plot
            (line_ret,) = ax_top.plot(
                [], [], color="#38BDF8", lw=2.0, label="Episode Return", zorder=4
            )
            (line_rolling_ret,) = ax_top.plot(
                [], [], color="#F59E0B", lw=1.8, ls="--", label="Rolling Return (20)", zorder=5
            )
            (line_reward,) = ax_top_right.plot(
                [],
                [],
                color="#34D399",
                lw=1.0,
                alpha=0.6,
                marker="o",
                markersize=2.5,
                label="Step Reward",
                zorder=3,
            )
            ax_top.axhline(0, color="#4B5563", lw=0.8, ls="--", alpha=0.5)

            # Legends for Top
            lines_top = [line_ret, line_rolling_ret, line_reward]
            labels_top = [l.get_label() for l in lines_top]
            ax_top.legend(
                lines_top,
                labels_top,
                loc="upper left",
                framealpha=0.75,
                facecolor="#1E293B",
                edgecolor="#334155",
                fontsize=8.5,
                labelcolor="#E2E8F0",
            )

            # Bottom: Policy Entropy
            ax_bot.set_facecolor("#111827")
            ax_bot.tick_params(colors="#9CA3AF", labelsize=9)
            ax_bot.set_xlabel("Environment Step", color="#9CA3AF", fontsize=10, fontweight="bold")
            ax_bot.set_ylabel("Policy Entropy (nats)", color="#C084FC", fontsize=10, fontweight="bold")
            ax_bot.grid(True, color="#1F2937", linestyle=":", alpha=0.7)

            (line_ent,) = ax_bot.plot(
                [], [], color="#C084FC", lw=1.5, alpha=0.7, label="Step Entropy", zorder=4
            )
            (line_rolling_ent,) = ax_bot.plot(
                [], [], color="#EC4899", lw=2.0, label="Rolling Entropy (20)", zorder=5
            )

            # Benchmark reference lines for Entropy
            ax_bot.axhline(
                self.max_entropy,
                color="#F43F5E",
                lw=1.2,
                ls="--",
                alpha=0.8,
                label=f"Max H (ln {self.n_actions} ≈ {self.max_entropy:.2f})",
            )
            ax_bot.axhline(0.3, color="#EF4444", lw=1.0, ls=":", alpha=0.6, label="Low Exploration (0.3)")

            ax_bot.legend(
                loc="upper right",
                framealpha=0.75,
                facecolor="#1E293B",
                edgecolor="#334155",
                fontsize=8.5,
                labelcolor="#E2E8F0",
            )

            for ax in (ax_top, ax_bot, ax_top_right):
                for spine in ax.spines.values():
                    spine.set_color("#374151")

            # HUD text banner across top
            hud_text = fig.suptitle(
                "Initializing TensorMew RL Monitor...",
                color="#F3F4F6",
                fontsize=11,
                fontweight="bold",
                y=0.98,
            )

            fig.tight_layout(rect=[0, 0.02, 1, 0.95])

            def on_close(event):
                self._closed = True

            fig.canvas.mpl_connect("close_event", on_close)

            self._fig = fig
            self._ax_top = ax_top
            self._ax_bot = ax_bot
            self._ax_top_right = ax_top_right
            self._line_ret = line_ret
            self._line_rolling_ret = line_rolling_ret
            self._line_reward = line_reward
            self._line_ent = line_ent
            self._line_rolling_ent = line_rolling_ent
            self._hud_text = hud_text
            self._initialized_gui = True

            plt.draw()
            plt.pause(0.001)
        except Exception:
            # Fall back to headless recording if display fails
            self.headless = True
            self._initialized_gui = False

    def log_step(
        self,
        *,
        reward: float,
        entropy: float,
        done: bool = False,
        action: int | str = "",
        value: float | None = None,
        info: Any = None,
        step: int | None = None,
        badges: int = 0,
        map_id: tuple[int, int] | None = None,
    ) -> StepRecord:
        """Record one environment step and update the live graph."""
        if not self.enabled:
            return StepRecord(
                step=self.total_steps,
                episode=self.current_episode,
                reward=reward,
                cumulative_return=self.current_ep_return + reward,
                entropy=entropy,
            )

        now = time.time()
        self.total_steps += 1
        current_step = self.total_steps if step is None else step

        self.current_ep_return += float(reward)

        disc_tile = False
        disc_map = False
        if info is not None:
            disc_tile = bool(getattr(info, "discovered_tile", False))
            disc_map = bool(getattr(info, "discovered_map", False))

        rec = StepRecord(
            step=current_step,
            episode=self.current_episode,
            reward=float(reward),
            cumulative_return=float(self.current_ep_return),
            entropy=float(entropy),
            action=action,
            value=float(value) if value is not None else None,
            discovered_tile=disc_tile,
            discovered_map=disc_map,
            badges=int(badges),
            map_id=map_id,
            timestamp=now,
        )
        self.step_history.append(rec)

        if self.jsonl_path is not None:
            self._append_jsonl(rec)

        if done:
            self.episode_returns.append(self.current_ep_return)
            self.current_episode += 1
            self.current_ep_return = 0.0

        if self._initialized_gui and not self._closed:
            if now - self._last_draw_time >= self.min_refresh_interval:
                self._draw_gui()
                self._last_draw_time = now

        return rec

    def log_update(
        self,
        *,
        update: int,
        mean_reward: float,
        mean_entropy: float | None = None,
        weight_change: float = 0.0,
        new_maps: int = 0,
    ) -> UpdateRecord:
        """Record one PPO update."""
        if not self.enabled:
            return UpdateRecord(
                update=update,
                mean_reward=mean_reward,
                mean_entropy=0.0,
                weight_change=weight_change,
            )

        now = time.time()
        if mean_entropy is None:
            recent_steps = self.step_history[-64:] if self.step_history else []
            mean_entropy = (
                float(np.mean([s.entropy for s in recent_steps])) if recent_steps else 0.0
            )

        rec = UpdateRecord(
            update=int(update),
            mean_reward=float(mean_reward),
            mean_entropy=float(mean_entropy),
            weight_change=float(weight_change),
            new_maps=int(new_maps),
            timestamp=now,
        )
        self.update_history.append(rec)

        if self._initialized_gui and not self._closed:
            self._draw_gui()

        return rec

    def _draw_gui(self) -> None:
        try:
            import matplotlib.pyplot as plt

            if not self._fig or self._closed or not plt.fignum_exists(self._fig.number):
                self._closed = True
                return

            if not self.step_history:
                return

            history = self.step_history
            if self.window_size is not None and len(history) > self.window_size:
                history = history[-self.window_size :]

            steps = [s.step for s in history]
            returns = [s.cumulative_return for s in history]
            rewards = [s.reward for s in history]
            entropies = [s.entropy for s in history]

            # Compute rolling averages
            roll_k = 20
            rolling_returns = self._rolling_mean(returns, k=roll_k)
            rolling_entropies = self._rolling_mean(entropies, k=roll_k)

            # Update line data
            self._line_ret.set_data(steps, returns)
            self._line_rolling_ret.set_data(steps, rolling_returns)
            self._line_reward.set_data(steps, rewards)
            self._line_ent.set_data(steps, entropies)
            self._line_rolling_ent.set_data(steps, rolling_entropies)

            # Update axis limits
            min_step, max_step = steps[0], steps[-1]
            if min_step == max_step:
                max_step += 1
            self._ax_top.set_xlim(min_step, max_step)
            self._ax_bot.set_xlim(min_step, max_step)

            # Top Y limits (Return & Reward)
            min_ret, max_ret = min(returns), max(returns)
            margin_ret = max(1.0, (max_ret - min_ret) * 0.15)
            self._ax_top.set_ylim(min_ret - margin_ret, max_ret + margin_ret)

            min_rew, max_rew = min(rewards), max(rewards)
            margin_rew = max(0.5, (max_rew - min_rew) * 0.2)
            self._ax_top_right.set_ylim(min_rew - margin_rew, max_rew + margin_rew)

            # Bottom Y limits (Entropy)
            max_ent = max(self.max_entropy * 1.05, max(entropies) + 0.1)
            self._ax_bot.set_ylim(0.0, max_ent)

            # HUD Update
            last = history[-1]
            last_roll_ret = rolling_returns[-1] if rolling_returns else last.cumulative_return
            last_roll_ent = rolling_entropies[-1] if rolling_entropies else last.entropy
            tiles_discovered = sum(1 for s in self.step_history if s.discovered_tile)
            maps_discovered = sum(1 for s in self.step_history if s.discovered_map)

            hud_msg = (
                f"Step: {last.step:<4d} | Ep: {last.episode} | Last Rew: {last.reward:+5.2f} | "
                f"Ep Return: {last.cumulative_return:+6.2f} (Roll: {last_roll_ret:+5.2f}) | "
                f"Entropy: {last.entropy:.3f} (Roll: {last_roll_ent:.3f}) | "
                f"Tiles: {tiles_discovered} | Maps: {maps_discovered}"
            )
            self._hud_text.set_text(hud_msg)

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception:
            self._closed = True

    @staticmethod
    def _rolling_mean(data: list[float], k: int = 20) -> list[float]:
        if not data:
            return []
        arr = np.array(data, dtype=np.float64)
        out = []
        for i in range(len(arr)):
            start = max(0, i - k + 1)
            out.append(float(np.mean(arr[start : i + 1])))
        return out

    def _append_jsonl(self, rec: StepRecord) -> None:
        try:
            line = json.dumps(asdict(rec))
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def save(self, path: str | Path | None = None) -> Path | None:
        """Render and save a high-resolution PNG snapshot of the curves."""
        dest = Path(path) if path else self.save_path
        if not dest or not self.step_history:
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            # If GUI figure already exists, save it directly
            if self._fig and not self._closed:
                self._fig.savefig(dest, dpi=160, facecolor=self._fig.get_facecolor())
                return dest

            # Otherwise, render fresh headless figure from step_history
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.5), facecolor="#0B0F19", sharex=True)
            ax1.set_facecolor("#111827")
            ax2.set_facecolor("#111827")
            for ax in (ax1, ax2):
                ax.tick_params(colors="#9CA3AF")
                ax.grid(True, color="#1F2937", linestyle=":", alpha=0.7)
                for sp in ax.spines.values():
                    sp.set_color("#374151")

            steps = [s.step for s in self.step_history]
            returns = [s.cumulative_return for s in self.step_history]
            rewards = [s.reward for s in self.step_history]
            entropies = [s.entropy for s in self.step_history]
            roll_ret = self._rolling_mean(returns, k=20)
            roll_ent = self._rolling_mean(entropies, k=20)

            ax1_r = ax1.twinx()
            ax1_r.tick_params(colors="#9CA3AF")
            ax1_r.set_ylabel("Step Reward", color="#34D399", fontweight="bold")
            for sp in ax1_r.spines.values():
                sp.set_color("#374151")

            l1, = ax1.plot(steps, returns, color="#38BDF8", lw=2.0, label="Episode Return")
            l2, = ax1.plot(steps, roll_ret, color="#F59E0B", lw=1.8, ls="--", label="Rolling Return (20)")
            l3, = ax1_r.plot(steps, rewards, color="#34D399", lw=1.0, alpha=0.6, marker="o", markersize=2.5, label="Step Reward")
            ax1.legend([l1, l2, l3], [l.get_label() for l in [l1, l2, l3]], loc="upper left", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
            ax1.set_ylabel("Episode Return", color="#38BDF8", fontweight="bold")

            ax2.plot(steps, entropies, color="#C084FC", lw=1.5, alpha=0.7, label="Step Entropy")
            ax2.plot(steps, roll_ent, color="#EC4899", lw=2.0, label="Rolling Entropy (20)")
            ax2.axhline(self.max_entropy, color="#F43F5E", lw=1.2, ls="--", label=f"Max H (ln {self.n_actions} ≈ {self.max_entropy:.2f})")
            ax2.legend(loc="upper right", facecolor="#1E293B", edgecolor="#334155", labelcolor="#E2E8F0")
            ax2.set_ylabel("Policy Entropy", color="#C084FC", fontweight="bold")
            ax2.set_xlabel("Environment Step", color="#9CA3AF", fontweight="bold")

            fig.suptitle(self.title, color="#F3F4F6", fontsize=12, fontweight="bold")
            fig.tight_layout()
            fig.savefig(dest, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)
            return dest
        except Exception:
            return None

    def close(self) -> None:
        """Close the live plot window."""
        self._closed = True
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
