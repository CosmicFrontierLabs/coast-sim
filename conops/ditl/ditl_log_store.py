"""Persistent storage for DITL logs using the Python standard library.

This module provides a lightweight SQLite-backed store for DITL events so you
can persist logs for many DITL runs without keeping everything in memory.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, PrivateAttr

from .ditl_event import DITLEvent


class DITLLogStore(BaseModel):
    """SQLite-backed store for DITL logs.

    The store schema is simple and optimized for querying by run and time.
    """

    db_path: Path = Field(default_factory=lambda: Path("ditl_logs.sqlite"))
    _conn: sqlite3.Connection = PrivateAttr()

    def __init__(self, db_path: str | Path = "ditl_logs.sqlite") -> None:  # type: ignore[override]
        # Allow convenient construction with str | Path while remaining a Pydantic model
        super().__init__(db_path=Path(db_path))
        self._conn = sqlite3.connect(self.db_path)
        # Improve concurrent read/write characteristics
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._create_schema()

    def __enter__(self) -> DITLLogStore:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                time REAL NOT NULL,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                description TEXT NOT NULL,
                obsid INTEGER,
                acs_mode INTEGER
            )
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_time ON events(run_id, time);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_run_type ON events(run_id, event_type);"
        )
        self._conn.commit()

    def add_event(self, run_id: str, event: DITLEvent) -> None:
        """Persist a single DITLEvent for a run_id."""
        self._conn.execute(
            """
            INSERT INTO events (run_id, time, timestamp, event_type, description, obsid, acs_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                float(event.time),
                event.timestamp,
                event.event_type,
                event.description,
                int(event.obsid) if event.obsid is not None else None,
                int(event.acs_mode) if event.acs_mode is not None else None,
            ),
        )
        self._conn.commit()

    def add_events(self, run_id: str, events: Iterable[DITLEvent]) -> None:
        """Persist multiple events efficiently in a transaction."""
        cur = self._conn.cursor()
        cur.executemany(
            """
            INSERT INTO events (run_id, time, timestamp, event_type, description, obsid, acs_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    float(ev.time),
                    ev.timestamp,
                    ev.event_type,
                    ev.description,
                    int(ev.obsid) if ev.obsid is not None else None,
                    int(ev.acs_mode) if ev.acs_mode is not None else None,
                )
                for ev in events
            ],
        )
        self._conn.commit()

    def fetch_events(
        self,
        run_id: str,
        start_time: float | None = None,
        end_time: float | None = None,
        event_type: str | None = None,
    ) -> list[DITLEvent]:
        """Fetch events for a run, optionally filtered by time range and type."""
        clauses: list[str] = ["run_id = ?"]
        args: list[object] = [run_id]

        if start_time is not None:
            clauses.append("time >= ?")
            args.append(float(start_time))
        if end_time is not None:
            clauses.append("time <= ?")
            args.append(float(end_time))
        if event_type is not None:
            clauses.append("event_type = ?")
            args.append(event_type)

        where = " AND ".join(clauses)
        cur = self._conn.cursor()
        cur.execute(
            f"SELECT time, timestamp, event_type, description, obsid, acs_mode FROM events WHERE {where} ORDER BY time ASC",
            args,
        )
        rows = cur.fetchall()
        events: list[DITLEvent] = []
        for time_val, timestamp, etype, desc, obsid, acs_mode in rows:
            # Rehydrate DITLEvent without reformatting timestamp
            events.append(
                DITLEvent(
                    time=float(time_val),
                    timestamp=str(timestamp),
                    event_type=str(etype),
                    description=str(desc),
                    obsid=int(obsid) if obsid is not None else None,
                    acs_mode=acs_mode if acs_mode is not None else None,
                )
            )
        return events

    def fetch_runs(self) -> list[str]:
        """Return distinct run_id values present in the store."""
        cur = self._conn.cursor()
        cur.execute("SELECT DISTINCT run_id FROM events ORDER BY run_id;")
        return [row[0] for row in cur.fetchall()]

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
