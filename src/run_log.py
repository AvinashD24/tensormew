"""Local SQLite log of segment runs, steps, and reasons."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DB = Path("logs/runs.db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    state_name TEXT,
    outcome TEXT,
    steps INTEGER,
    start_map_group INTEGER,
    start_map_number INTEGER,
    end_map_group INTEGER,
    end_map_number INTEGER,
    start_badges INTEGER,
    end_badges INTEGER,
    cut_cleared INTEGER,
    rom_sha1 TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    timestamp TEXT,
    frame INTEGER,
    map_group INTEGER,
    map_number INTEGER,
    x INTEGER,
    y INTEGER,
    badges INTEGER,
    actor TEXT,
    trigger TEXT,
    tool TEXT,
    arguments TEXT,
    reason TEXT,
    ok INTEGER,
    est_usd REAL,
    input_tokens INTEGER,
    output_tokens INTEGER
);

CREATE INDEX IF NOT EXISTS steps_run_id ON steps(run_id);
CREATE INDEX IF NOT EXISTS steps_ok ON steps(ok);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunLog:
    def __init__(self, path: str | Path = DEFAULT_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(_SCHEMA)
        self._migrate()
        self.conn.commit()

    def _migrate(self) -> None:
        columns = {row[1] for row in self.conn.execute("PRAGMA table_info(runs)")}
        if "checkpoint_name" not in columns:
            self.conn.execute("ALTER TABLE runs ADD COLUMN checkpoint_name TEXT")

    def close(self) -> None:
        self.conn.close()

    def start_run(
        self,
        *,
        state_name: str = "",
        start_map: tuple[int, int] = (0, 0),
        start_badges: int = 0,
        rom_sha1: str = "",
    ) -> int:
        cur = self.conn.execute(
            """
            INSERT INTO runs (
                started_at, state_name, start_map_group, start_map_number,
                start_badges, rom_sha1, outcome, steps
            ) VALUES (?, ?, ?, ?, ?, ?, 'running', 0)
            """,
            (_now(), state_name, start_map[0], start_map[1], start_badges, rom_sha1),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def add_step(self, run_id: int, row: dict) -> None:
        map_pair = row.get("map") or [None, None]
        pos = row.get("pos") or [None, None]
        self.conn.execute(
            """
            INSERT INTO steps (
                run_id, timestamp, frame, map_group, map_number, x, y, badges,
                actor, trigger, tool, arguments, reason, ok, est_usd,
                input_tokens, output_tokens
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                row.get("timestamp") or _now(),
                row.get("frame"),
                map_pair[0],
                map_pair[1],
                pos[0],
                pos[1],
                row.get("badges"),
                row.get("actor"),
                row.get("trigger") or "",
                row.get("tool"),
                json.dumps(row.get("arguments") or {}),
                row.get("reason") or "",
                1 if row.get("ok") else 0,
                float(row.get("est_usd") or 0),
                int(row.get("input_tokens") or 0),
                int(row.get("output_tokens") or 0),
            ),
        )
        self.conn.commit()

    def finish_run(
        self,
        run_id: int,
        *,
        outcome: str,
        end_map: tuple[int, int],
        end_badges: int,
        steps: int,
        cut_cleared: bool,
        error: str = "",
    ) -> None:
        self.conn.execute(
            """
            UPDATE runs SET
                ended_at=?, outcome=?, steps=?,
                end_map_group=?, end_map_number=?, end_badges=?,
                cut_cleared=?, error=?
            WHERE id=?
            """,
            (
                _now(),
                outcome,
                steps,
                end_map[0] if end_map else None,
                end_map[1] if len(end_map) > 1 else None,
                end_badges,
                1 if cut_cleared else 0,
                error,
                run_id,
            ),
        )
        self.conn.commit()

    def set_checkpoint(self, run_id: int, name: str) -> None:
        self.conn.execute(
            "UPDATE runs SET checkpoint_name=? WHERE id=?",
            (name, run_id),
        )
        self.conn.commit()

    def recent_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                (limit,),
            )
        )

    def failed_steps(self, limit: int = 50) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                """
                SELECT steps.*, runs.state_name, runs.outcome AS run_outcome
                FROM steps JOIN runs ON runs.id = steps.run_id
                WHERE steps.ok = 0
                ORDER BY steps.id DESC
                LIMIT ?
                """,
                (limit,),
            )
        )

    def run_steps(self, run_id: int) -> list[sqlite3.Row]:
        return list(
            self.conn.execute(
                "SELECT * FROM steps WHERE run_id=? ORDER BY id",
                (run_id,),
            )
        )

    def progress(self) -> dict:
        row = self.conn.execute(
            """
            SELECT
                COUNT(*) AS runs,
                COALESCE(MAX(end_badges), 0) AS max_badges,
                SUM(CASE WHEN outcome='advanced' THEN 1 ELSE 0 END) AS advanced,
                SUM(CASE WHEN outcome='error' THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN outcome='stuck' THEN 1 ELSE 0 END) AS stuck
            FROM runs
            WHERE outcome IS NOT NULL AND outcome != 'running'
            """
        ).fetchone()
        return {
            "runs": int(row["runs"] or 0),
            "max_badges": int(row["max_badges"] or 0),
            "advanced": int(row["advanced"] or 0),
            "errors": int(row["errors"] or 0),
            "stuck": int(row["stuck"] or 0),
        }
