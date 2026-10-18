"""Visual RL training dashboard for live PyBoy runs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.rl_env import ACTIONS, N_ACTIONS

ACTION_LABELS = list(ACTIONS)


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
    approx_kl: float = 0.0
    clip_frac: float = 0.0
    value_mse: float = 0.0
    max_badges: int = 0
    unique_maps: int = 0
    timestamp: float = 0.0


class LiveMetricsPlotter:
    """Live Matplotlib dashboard: episodes, PPO updates, entropy, game progress, action mix."""

    def __init__(
        self,
        *,
        title: str = "TensorMew RL Monitor",
        enabled: bool = True,
        headless: bool = False,
        window_size: int | None = 512,
        refresh_fps: float = 8.0,
        save_path: str | Path | None = "logs/reward_entropy_curve.png",
        jsonl_path: str | Path | None = "logs/metrics_history.jsonl",
        n_actions: int = N_ACTIONS,
        lr: float = 0.05,
        clip: float = 0.2,
        entropy_coef: float = 0.01,
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
        self.hyperparams = {"lr": lr, "clip": clip, "entropy_coef": entropy_coef}

        self.step_history: list[StepRecord] = []
        self.update_history: list[UpdateRecord] = []
        self.episode_returns: list[float] = []

        self.current_episode = 0
        self.current_ep_return = 0.0
        self.total_steps = 0
        self.max_badges = 0
        self.unique_maps: set[tuple[int, int]] = set()
        self._rollout_action_counts = np.zeros(n_actions, dtype=np.int64)
        self._last_probs: np.ndarray | None = None

        self._last_draw_time = 0.0
        self._fig = None
        self._axes: dict[str, Any] = {}
        self._lines: dict[str, Any] = {}
        self._hud_text = None
        self._action_bars: list[Any] = []
        self._closed = False
        self._initialized_gui = False
        self._jsonl_buffer: list[str] = []

        if self.enabled and not self.headless:
            self._init_gui()

        if self.jsonl_path is not None:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _init_gui(self) -> None:
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, axes = plt.subplots(
                2,
                2,
                figsize=(11.0, 7.5),
                facecolor="#0B0F19",
                gridspec_kw={"hspace": 0.32, "wspace": 0.28},
            )
            fig.canvas.manager.set_window_title(self.title)

            ax_ep, ax_up = axes[0]
            ax_ent, ax_prog = axes[1]

            for ax in (ax_ep, ax_up, ax_ent, ax_prog):
                ax.set_facecolor("#111827")
                ax.tick_params(colors="#9CA3AF", labelsize=8)
                ax.grid(True, color="#1F2937", linestyle=":", alpha=0.7)
                for spine in ax.spines.values():
                    spine.set_color("#374151")

            ax_ep.set_title("Episode return", color="#E5E7EB", fontsize=10, loc="left")
            ax_ep.set_xlabel("Episode", color="#9CA3AF", fontsize=9)
            ax_ep.set_ylabel("Return", color="#38BDF8", fontsize=9, fontweight="bold")

            ax_up.set_title("PPO update (batch mean)", color="#E5E7EB", fontsize=10, loc="left")
            ax_up.set_xlabel("Update", color="#9CA3AF", fontsize=9)
            ax_up.set_ylabel("Mean reward", color="#34D399", fontsize=9, fontweight="bold")
            ax_up_r = ax_up.twinx()
            ax_up_r.set_ylabel("|Δw|", color="#F59E0B", fontsize=9, fontweight="bold")
            ax_up_r.tick_params(colors="#9CA3AF", labelsize=8)
            for spine in ax_up_r.spines.values():
                spine.set_color("#374151")

            ax_ent.set_title("Policy entropy", color="#E5E7EB", fontsize=10, loc="left")
            ax_ent.set_xlabel("Update", color="#9CA3AF", fontsize=9)
            ax_ent.set_ylabel("Entropy (nats)", color="#C084FC", fontsize=9, fontweight="bold")

            ax_prog.set_title("Game progress", color="#E5E7EB", fontsize=10, loc="left")
            ax_prog.set_xlabel("Update", color="#9CA3AF", fontsize=9)
            ax_prog.set_ylabel("Badges", color="#F472B6", fontsize=9, fontweight="bold")
            ax_prog_r = ax_prog.twinx()
            ax_prog_r.set_ylabel("Unique maps", color="#60A5FA", fontsize=9, fontweight="bold")
            ax_prog_r.tick_params(colors="#9CA3AF", labelsize=8)
            for spine in ax_prog_r.spines.values():
                spine.set_color("#374151")

            (line_ep,) = ax_ep.plot([], [], color="#38BDF8", lw=1.5, marker="o", markersize=4, label="Return")
            (line_ep_roll,) = ax_ep.plot(
                [], [], color="#F59E0B", lw=1.8, ls="--", label="Roll (10 ep)"
            )

            (line_mean_rew,) = ax_up.plot([], [], color="#34D399", lw=2.0, marker="s", markersize=3, label="Mean reward")
            (line_wchg,) = ax_up_r.plot([], [], color="#F59E0B", lw=1.5, ls="--", label="|Δw|")

            (line_ent_up,) = ax_ent.plot([], [], color="#EC4899", lw=2.0, marker="o", markersize=3, label="Batch mean H")
            ax_ent.axhline(
                self.max_entropy,
                color="#F43F5E",
                lw=1.0,
                ls="--",
                alpha=0.75,
                label=f"Max H ({self.max_entropy:.2f})",
            )
            ax_ent.axhline(0.3, color="#EF4444", lw=0.9, ls=":", alpha=0.55, label="Low H (0.3)")
            ax_ent.legend(
                loc="upper right",
                framealpha=0.75,
                facecolor="#1E293B",
                edgecolor="#334155",
                fontsize=7.5,
                labelcolor="#E2E8F0",
            )

            (line_badges,) = ax_prog.plot([], [], color="#F472B6", lw=2.0, marker="o", markersize=4, label="Badges")
            (line_maps,) = ax_prog_r.plot(
                [], [], color="#60A5FA", lw=1.8, ls="--", marker="^", markersize=3, label="Unique maps"
            )
            (line_frontier,) = ax_prog_r.plot(
                [], [], color="#A78BFA", lw=1.2, ls=":", marker="x", markersize=4, label="Frontier (+maps)"
            )

            ax_ep.legend(
                loc="upper left",
                framealpha=0.75,
                facecolor="#1E293B",
                edgecolor="#334155",
                fontsize=8,
                labelcolor="#E2E8F0",
            )

            # Action mix bars (last rollout)
            bar_ax = fig.add_axes([0.08, 0.02, 0.84, 0.06], facecolor="#111827")
            bar_ax.set_xlim(-0.5, self.n_actions - 0.5)
            bar_ax.set_ylim(0, 1)
            bar_ax.set_xticks(range(self.n_actions))
            bar_ax.set_xticklabels(ACTION_LABELS[: self.n_actions], color="#9CA3AF", fontsize=8)
            bar_ax.tick_params(axis="y", colors="#9CA3AF", labelsize=7)
            bar_ax.set_ylabel("p(action)", color="#9CA3AF", fontsize=7)
            for spine in bar_ax.spines.values():
                spine.set_color("#374151")
            action_bars = bar_ax.bar(
                range(self.n_actions),
                [1.0 / self.n_actions] * self.n_actions,
                color="#6366F1",
                alpha=0.85,
                width=0.65,
            )

            hud = fig.suptitle(
                "Initializing…",
                color="#F3F4F6",
                fontsize=10,
                fontweight="bold",
                y=0.98,
            )

            def on_close(_event):
                self._closed = True

            fig.canvas.mpl_connect("close_event", on_close)

            self._fig = fig
            self._axes = {
                "ep": ax_ep,
                "up": ax_up,
                "up_r": ax_up_r,
                "ent": ax_ent,
                "prog": ax_prog,
                "prog_r": ax_prog_r,
                "bars": bar_ax,
            }
            self._lines = {
                "ep": line_ep,
                "ep_roll": line_ep_roll,
                "mean_rew": line_mean_rew,
                "wchg": line_wchg,
                "ent_up": line_ent_up,
                "badges": line_badges,
                "maps": line_maps,
                "frontier": line_frontier,
            }
            self._action_bars = list(action_bars)
            self._hud_text = hud
            self._initialized_gui = True

            plt.draw()
            plt.pause(0.001)
        except Exception:
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
        probabilities: np.ndarray | None = None,
    ) -> StepRecord:
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

        badges_i = int(badges)
        self.max_badges = max(self.max_badges, badges_i)
        if map_id is not None:
            self.unique_maps.add(map_id)

        action_idx = action if isinstance(action, int) else 0
        if isinstance(action, int) and 0 <= action < self.n_actions:
            self._rollout_action_counts[action_idx] += 1

        if probabilities is not None:
            self._last_probs = np.asarray(probabilities, dtype=np.float64)

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
            badges=badges_i,
            map_id=map_id,
            timestamp=now,
        )
        self.step_history.append(rec)
        self._append_jsonl({"type": "step", **asdict(rec)})

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
        approx_kl: float = 0.0,
        clip_frac: float = 0.0,
        value_mse: float = 0.0,
    ) -> UpdateRecord:
        if not self.enabled:
            return UpdateRecord(
                update=update,
                mean_reward=mean_reward,
                mean_entropy=0.0,
                weight_change=weight_change,
            )

        now = time.time()
        if mean_entropy is None:
            recent = self.step_history[-64:] if self.step_history else []
            mean_entropy = float(np.mean([s.entropy for s in recent])) if recent else 0.0

        rec = UpdateRecord(
            update=int(update),
            mean_reward=float(mean_reward),
            mean_entropy=float(mean_entropy),
            weight_change=float(weight_change),
            new_maps=int(new_maps),
            approx_kl=float(approx_kl),
            clip_frac=float(clip_frac),
            value_mse=float(value_mse),
            max_badges=self.max_badges,
            unique_maps=len(self.unique_maps),
            timestamp=now,
        )
        self.update_history.append(rec)
        self._append_jsonl({"type": "update", **asdict(rec)})
        self._rollout_action_counts[:] = 0

        if self._initialized_gui and not self._closed:
            self._draw_gui()

        return rec

    def _draw_gui(self) -> None:
        try:
            import matplotlib.pyplot as plt

            if not self._fig or self._closed or not plt.fignum_exists(self._fig.number):
                self._closed = True
                return

            # Episode panel
            if self.episode_returns:
                eps = list(range(len(self.episode_returns)))
                rets = self.episode_returns
                roll = self._rolling_mean(rets, k=10)
                self._lines["ep"].set_data(eps, rets)
                self._lines["ep_roll"].set_data(eps, roll)
                y0, y1 = self._percentile_ylim(rets, pad=0.12, floor=1.0)
                self._axes["ep"].set_xlim(-0.5, max(len(eps) - 0.5, 0.5))
                self._axes["ep"].set_ylim(y0, y1)

            # Update-indexed panels
            if self.update_history:
                ups = [u.update for u in self.update_history]
                self._lines["mean_rew"].set_data(ups, [u.mean_reward for u in self.update_history])
                self._lines["wchg"].set_data(ups, [u.weight_change for u in self.update_history])
                self._lines["ent_up"].set_data(ups, [u.mean_entropy for u in self.update_history])
                self._lines["badges"].set_data(ups, [u.max_badges for u in self.update_history])
                self._lines["maps"].set_data(ups, [u.unique_maps for u in self.update_history])
                self._lines["frontier"].set_data(ups, [u.new_maps for u in self.update_history])

                x_max = max(ups[-1], 1)
                for key in ("up", "ent", "prog"):
                    self._axes[key].set_xlim(-0.5, x_max + 0.5)

                rews = [u.mean_reward for u in self.update_history]
                y0, y1 = self._percentile_ylim(rews, pad=0.15, floor=0.05)
                self._axes["up"].set_ylim(y0, y1)
                wchg = [u.weight_change for u in self.update_history]
                w0, w1 = self._percentile_ylim(wchg, pad=0.2, floor=1e-4)
                self._axes["up_r"].set_ylim(w0, w1)

                self._axes["ent"].set_ylim(0.0, max(self.max_entropy * 1.05, 0.5))

                badges = [u.max_badges for u in self.update_history]
                maps = [u.unique_maps for u in self.update_history]
                self._axes["prog"].set_ylim(-0.2, max(1.0, max(badges) + 0.5))
                self._axes["prog_r"].set_ylim(0, max(2, max(maps) + 1))

            # Action probability bars
            probs = self._last_probs
            if probs is None or len(probs) != self.n_actions:
                probs = np.ones(self.n_actions) / self.n_actions
            for bar, p in zip(self._action_bars, probs, strict=True):
                bar.set_height(float(p))

            last = self.step_history[-1] if self.step_history else None
            last_up = self.update_history[-1] if self.update_history else None
            hp = self.hyperparams
            hud_parts = [
                f"Step {self.total_steps}",
                f"Ep {self.current_episode}",
            ]
            if last:
                hud_parts.append(f"rew {last.reward:+.2f}")
                hud_parts.append(f"H {last.entropy:.2f}")
                if last.map_id:
                    hud_parts.append(f"map {last.map_id[0]},{last.map_id[1]}")
                hud_parts.append(f"badges {last.badges}")
            if last_up:
                hud_parts.append(
                    f"upd {last_up.update} μR {last_up.mean_reward:+.3f} KL {last_up.approx_kl:.4f} clip {last_up.clip_frac:.2f}"
                )
            hud_parts.append(f"lr={hp['lr']} clip={hp['clip']} ent={hp['entropy_coef']}")
            self._hud_text.set_text("  |  ".join(hud_parts))

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

    @staticmethod
    def _percentile_ylim(values: list[float], *, pad: float, floor: float) -> tuple[float, float]:
        if not values:
            return (-1.0, 1.0)
        arr = np.asarray(values, dtype=np.float64)
        lo = float(np.percentile(arr, 5))
        hi = float(np.percentile(arr, 95))
        if lo == hi:
            lo -= floor
            hi += floor
        margin = max(floor, (hi - lo) * pad)
        return lo - margin, hi + margin

    def _append_jsonl(self, payload: dict) -> None:
        if self.jsonl_path is None:
            return
        try:
            line = json.dumps(payload)
            self._jsonl_buffer.append(line)
            if len(self._jsonl_buffer) >= 32:
                self._flush_jsonl()
        except Exception:
            pass

    def _flush_jsonl(self) -> None:
        if not self._jsonl_buffer or self.jsonl_path is None:
            return
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write("\n".join(self._jsonl_buffer) + "\n")
            self._jsonl_buffer.clear()
        except Exception:
            pass

    def save(self, path: str | Path | None = None) -> Path | None:
        dest = Path(path) if path else self.save_path
        if not dest or (not self.step_history and not self.update_history):
            return None

        self._flush_jsonl()
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            import matplotlib.pyplot as plt

            if self._fig is not None and not self._closed:
                try:
                    self._draw_gui()
                    if plt.fignum_exists(self._fig.number):
                        self._fig.savefig(dest, dpi=160, facecolor=self._fig.get_facecolor())
                        return dest
                except Exception:
                    pass

            fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), facecolor="#0B0F19")
            self._render_static(fig, axes)
            fig.savefig(dest, dpi=160, facecolor=fig.get_facecolor())
            plt.close(fig)
            return dest
        except Exception:
            return None

    def _render_static(self, fig, axes) -> None:
        ax_ep, ax_up = axes[0]
        ax_ent, ax_prog = axes[1]
        for ax in (ax_ep, ax_up, ax_ent, ax_prog):
            ax.set_facecolor("#111827")
            ax.tick_params(colors="#9CA3AF", labelsize=8)
            ax.grid(True, color="#1F2937", linestyle=":", alpha=0.7)

        if self.episode_returns:
            eps = list(range(len(self.episode_returns)))
            ax_ep.plot(eps, self.episode_returns, color="#38BDF8", marker="o", markersize=3)
            ax_ep.plot(eps, self._rolling_mean(self.episode_returns, 10), color="#F59E0B", ls="--")
        ax_ep.set_title("Episode return", color="#E5E7EB")

        if self.update_history:
            ups = [u.update for u in self.update_history]
            ax_up.plot(ups, [u.mean_reward for u in self.update_history], color="#34D399", marker="s", markersize=3)
            ax_up_r = ax_up.twinx()
            ax_up_r.plot(ups, [u.weight_change for u in self.update_history], color="#F59E0B", ls="--")
            ax_ent.plot(ups, [u.mean_entropy for u in self.update_history], color="#EC4899", marker="o", markersize=3)
            ax_ent.axhline(self.max_entropy, color="#F43F5E", ls="--", alpha=0.7)
            ax_prog.plot(ups, [u.max_badges for u in self.update_history], color="#F472B6", marker="o")
            ax_prog_r = ax_prog.twinx()
            ax_prog_r.plot(ups, [u.unique_maps for u in self.update_history], color="#60A5FA", ls="--")

        ax_up.set_title("Mean reward / |Δw|", color="#E5E7EB")
        ax_ent.set_title("Entropy", color="#E5E7EB")
        ax_prog.set_title("Badges / maps", color="#E5E7EB")
        fig.suptitle(self.title, color="#F3F4F6", fontsize=11)
        fig.tight_layout()

    def close(self) -> None:
        self._flush_jsonl()
        self._closed = True
        if self._fig is not None:
            try:
                import matplotlib.pyplot as plt

                plt.close(self._fig)
            except Exception:
                pass
            self._fig = None
