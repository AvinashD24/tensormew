"""PokéMind thought log: read logs/decisions.jsonl and show why each step ran."""

from __future__ import annotations

import html
import json
from pathlib import Path

DEFAULT_LOG = Path("logs/decisions.jsonl")
EMPTY_HINT = "python -m tools.run_segment --state ilex_src"
RECENT_LIMIT = 50


def load_decisions(path: str | Path) -> list[dict]:
    """Read a decisions JSONL file. Missing files yield []. Bad lines are skipped."""
    log_path = Path(path)
    if not log_path.is_file():
        return []
    rows: list[dict] = []
    with log_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


def summarize(rows: list[dict]) -> dict:
    """Running totals for the thought log. `badges` is the latest row's count."""
    est_usd = 0.0
    tokens = 0
    for row in rows:
        est_usd += _as_float(row.get("est_usd"))
        tokens += _as_int(row.get("input_tokens")) + _as_int(row.get("output_tokens"))
    latest = rows[-1] if rows else {}
    return {
        "steps": len(rows),
        "est_usd": est_usd,
        "tokens": tokens,
        "latest_reason": str(latest.get("reason") or ""),
        "badges": latest.get("badges", 0) if rows else 0,
    }


def recent_table(rows: list[dict], limit: int = RECENT_LIMIT) -> list[dict]:
    """Last `limit` decisions as flat display rows (oldest of the window first)."""
    table: list[dict] = []
    for row in rows[-limit:]:
        table.append(
            {
                "time": row.get("timestamp", ""),
                "map": _fmt_pair(row.get("map")),
                "pos": _fmt_pair(row.get("pos")),
                "badges": row.get("badges", ""),
                "actor": row.get("actor", ""),
                "trigger": row.get("trigger", ""),
                "tool": row.get("tool", ""),
                "reason": row.get("reason", ""),
                "ok": row.get("ok", ""),
                "est_usd": row.get("est_usd", 0),
            }
        )
    return table


def _fmt_pair(value) -> str:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return f"{value[0]},{value[1]}"
    if value is None:
        return ""
    return str(value)


def _as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def extract_metric_series(rows: list[dict]) -> dict:
    """Extract return and entropy progression from decision rows."""
    steps: list[int] = []
    rewards: list[float] = []
    returns: list[float] = []
    entropies: list[float] = []
    running_return = 0.0
    for idx, row in enumerate(rows):
        if "reward" in row or "entropy" in row:
            steps.append(idx + 1)
            rew = float(row.get("reward", 0.0))
            running_return += rew
            rewards.append(rew)
            returns.append(running_return)
            entropies.append(float(row.get("entropy", 0.0)))
    return {
        "step": steps,
        "reward": rewards,
        "cumulative_return": returns,
        "entropy": entropies,
    }


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="PokéMind", layout="wide")
    st.title("PokéMind")

    with st.sidebar:
        st.header("Decision log")
        path = st.text_input("JSONL path", value=str(DEFAULT_LOG))

    rows = load_decisions(path)
    summary = summarize(rows)
    if not rows:
        st.info("No decisions yet. The log is missing or empty.")
        st.markdown("Run a segment to write the thought log:")
        st.code(EMPTY_HINT, language="bash")
        return

    st.subheader("Thought log")
    with st.container(border=True):
        st.markdown(
            (
                "<p style='font-size:1.75rem;line-height:1.35;margin:0'>"
                f"{html.escape(summary['latest_reason'])}"
                "</p>"
            ),
            unsafe_allow_html=True,
        )

    steps, cost, tokens, badges = st.columns(4)
    steps.metric("Steps", summary["steps"])
    cost.metric("Running cost", f"${summary['est_usd']:.6f}")
    tokens.metric("Tokens", f"{summary['tokens']:,}")
    badges.metric("Badges", summary["badges"])

    st.subheader("Recent decisions")
    st.caption(f"Last {min(len(rows), RECENT_LIMIT)} of {summary['steps']} steps.")
    st.dataframe(recent_table(rows), use_container_width=True, hide_index=True)

    metrics = extract_metric_series(rows)
    if metrics["step"]:
        st.subheader("Model Returns & Policy Entropy")
        col1, col2 = st.columns(2)
        with col1:
            st.caption("Cumulative Model Return & Step Reward")
            st.line_chart(
                {"Cumulative Return": metrics["cumulative_return"], "Step Reward": metrics["reward"]}
            )
        with col2:
            st.caption("Policy Action Entropy")
            st.line_chart({"Entropy": metrics["entropy"]})

    for plot_candidate in (
        Path("logs/training_reward_entropy.png"),
        Path("logs/segment_reward_entropy.png"),
        Path("logs/reward_entropy_curve.png"),
    ):
        if plot_candidate.is_file():
            st.subheader(f"Saved Run Curve ({plot_candidate.name})")
            st.image(str(plot_candidate), use_container_width=True)
            break


if __name__ == "__main__":
    main()
