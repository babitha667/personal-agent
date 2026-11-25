"""Basic action persistence; worker claims and scheduling live in action_queue.py."""

from dataclasses import asdict
from pathlib import Path

from database import connect, initialize_task_tables
from models import Action
from action_data import action_record, action_from_record


class ActionStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        initialize_task_tables(db_path)

    def create(self, action: Action) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO actions
                (action_id, task_id, exec_at, description, status, attempt_count, created_at, context)
                VALUES (:action_id, :task_id, :exec_at, :description, :status,
                        :attempt_count, :created_at, :context)""",
                action_record(action),
            )

    def get(self, action_id: str) -> Action | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM actions WHERE action_id = ?", (action_id,)
            ).fetchone()
        return action_from_record(row) if row else None
