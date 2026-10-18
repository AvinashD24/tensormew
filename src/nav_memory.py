"""Remember map changes the agent actually walked. No stored routes."""

from __future__ import annotations

import sqlite3
from pathlib import Path


class NavMemory:
    def __init__(self, path: str | Path = "logs/nav.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS transitions (
                src_group INTEGER NOT NULL,
                src_number INTEGER NOT NULL,
                src_x INTEGER NOT NULL,
                src_y INTEGER NOT NULL,
                direction TEXT NOT NULL,
                dst_group INTEGER NOT NULL,
                dst_number INTEGER NOT NULL,
                dst_x INTEGER NOT NULL,
                dst_y INTEGER NOT NULL,
                seen_count INTEGER NOT NULL DEFAULT 1,
                PRIMARY KEY (src_group, src_number, src_x, src_y, direction, dst_group, dst_number)
            )
            """
        )
        self.conn.commit()

    def record(
        self,
        src_map: tuple[int, int],
        src_xy: tuple[int, int],
        direction: str,
        dst_map: tuple[int, int],
        dst_xy: tuple[int, int],
    ) -> None:
        if tuple(src_map) == tuple(dst_map):
            return
        self.conn.execute(
            """
            INSERT INTO transitions (
                src_group, src_number, src_x, src_y, direction,
                dst_group, dst_number, dst_x, dst_y, seen_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT (src_group, src_number, src_x, src_y, direction, dst_group, dst_number)
            DO UPDATE SET seen_count = seen_count + 1,
                          dst_x = excluded.dst_x,
                          dst_y = excluded.dst_y
            """,
            (
                src_map[0],
                src_map[1],
                src_xy[0],
                src_xy[1],
                direction,
                dst_map[0],
                dst_map[1],
                dst_xy[0],
                dst_xy[1],
            ),
        )
        self.conn.commit()

    def exits_from(self, map_id: tuple[int, int]) -> list[dict]:
        rows = self.conn.execute(
            """
            SELECT src_x, src_y, direction, dst_group, dst_number, dst_x, dst_y, seen_count
            FROM transitions
            WHERE src_group=? AND src_number=?
            ORDER BY seen_count DESC, direction
            """,
            (map_id[0], map_id[1]),
        )
        return [
            {
                "src_xy": (row["src_x"], row["src_y"]),
                "direction": row["direction"],
                "dst_map": (row["dst_group"], row["dst_number"]),
                "dst_xy": (row["dst_x"], row["dst_y"]),
                "seen_count": row["seen_count"],
            }
            for row in rows
        ]

    def close(self) -> None:
        self.conn.close()
