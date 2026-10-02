"""Versioned warehouse column contract."""

from __future__ import annotations

import duckdb

CONTRACT_VERSION = "1"

INTERACTION_COLUMNS = [
    "user_id",
    "item_id",
    "timestamp",
    "session_id",
    "feedback_type",
    "value",
]
ITEM_COLUMNS = ["item_id", "text", "category", "image_path"]
USER_COLUMNS = ["user_id", "attributes", "group_label"]


class ContractError(Exception):
    pass


def assert_columns(con: duckdb.DuckDBPyConnection, table: str, expected: list[str]) -> None:
    rows = con.execute(f"DESCRIBE {table}").fetchall()
    have = [row[0] for row in rows]
    missing = [col for col in expected if col not in have]
    if missing:
        raise ContractError(
            f"{table} is missing {missing}. Contract version {CONTRACT_VERSION} requires {expected}."
        )
