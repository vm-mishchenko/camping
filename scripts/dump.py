"""Inspect checkpointed threads as readable JSON.

Usage:
  python scripts/dump.py                    # list all thread_ids in the db
  python scripts/dump.py <thread_id>        # dump latest state for one thread
  python scripts/dump.py <thread_id> --all  # dump full checkpoint history
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

CHECKPOINT_DB = Path(__file__).resolve().parent.parent / "data" / "checkpoints.db"


def list_threads() -> list[str]:
    if not CHECKPOINT_DB.exists():
        return []
    conn = sqlite3.connect(str(CHECKPOINT_DB))
    try:
        rows = conn.execute("SELECT DISTINCT thread_id FROM checkpoints").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


async def dump_thread(thread_id: str, all_checkpoints: bool) -> None:
    config = {"configurable": {"thread_id": thread_id}}
    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_DB)) as saver:
        if all_checkpoints:
            checkpoints = []
            async for cp in saver.alist(config):
                checkpoints.append({
                    "checkpoint_id": cp.config["configurable"].get("checkpoint_id"),
                    "metadata": cp.metadata,
                    "state": cp.checkpoint["channel_values"],
                    "next_nodes": list(getattr(cp, "next", []) or []),
                })
            print(json.dumps(checkpoints, indent=2, default=str))
        else:
            cp = await saver.aget_tuple(config)
            if cp is None:
                print(f"No checkpoints for thread_id={thread_id}")
                return
            print(json.dumps({
                "thread_id": thread_id,
                "checkpoint_id": cp.config["configurable"].get("checkpoint_id"),
                "metadata": cp.metadata,
                "state": cp.checkpoint["channel_values"],
                "next_nodes": list(getattr(cp, "next", []) or []),
            }, indent=2, default=str))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump checkpointer state as JSON.")
    parser.add_argument("thread_id", nargs="?", help="Thread to dump. If omitted, list all threads.")
    parser.add_argument("--all", action="store_true", help="Dump full checkpoint history, not just latest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.thread_id:
        threads = list_threads()
        if not threads:
            print(f"No threads in {CHECKPOINT_DB}")
            return
        print(f"Threads in {CHECKPOINT_DB}:")
        for t in threads:
            print(f"  {t}")
        return
    asyncio.run(dump_thread(args.thread_id, args.all))


if __name__ == "__main__":
    main()
