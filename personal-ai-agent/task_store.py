"""Task persistence, independent of action storage."""

from dataclasses import asdict
from pathlib import Path

from database import connect, initialize_task_tables
from models import Task
from action_data import parse_context


class TaskStore:
    def __init__(self, db_path: Path, *, initialize: bool = True) -> None:
        self.db_path = db_path
        if initialize:
            initialize_task_tables(db_path)

    def create(self, task: Task) -> None:
        with connect(self.db_path) as connection:
            connection.execute(
                """INSERT INTO tasks
                (task_id, user_id, description, status, period, created_at, last_executed_at)
                VALUES (:task_id, :user_id, :description, :status, :period,
                        :created_at, :last_executed_at)""",
                asdict(task),
            )

    def get(self, task_id: str) -> Task | None:
        with connect(self.db_path) as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return Task(**dict(row)) if row else None

    def user_context(self, user_id: str, *, read_only: bool = False) -> list[dict]:
        """Retrieve all tasks and associated actions belonging to this user."""
        with connect(self.db_path, read_only=read_only) as connection:
            tasks = [dict(row) for row in connection.execute(
                "SELECT * FROM tasks WHERE user_id = ? ORDER BY created_at, task_id",
                (user_id,),
            )]
            has_results=connection.execute("SELECT 1 FROM sqlite_master WHERE name='action_results'").fetchone()
            for task in tasks:
                task["actions"] = [dict(row, context=parse_context(row["context"])) for row in connection.execute(
                    "SELECT * FROM actions WHERE task_id = ? ORDER BY created_at, action_id",
                    (task["task_id"],),
                )]
                if has_results:
                    row=connection.execute('''SELECT r.* FROM action_results r JOIN actions a USING(action_id)
                        WHERE a.task_id=? ORDER BY r.completed_at DESC LIMIT 1''',(task['task_id'],)).fetchone()
                    task['last_result']=dict(row,details=parse_context(row['details']) if 'details' in row.keys() else {}) if row else None
        return tasks
