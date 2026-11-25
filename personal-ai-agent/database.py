"""Shared SQLite connection and schema initialization."""

import sqlite3
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def connect(db_path: Path, *, read_only: bool = False) -> Iterator[sqlite3.Connection]:
    if read_only:
        connection = sqlite3.connect(Path(db_path).resolve().as_uri() + '?mode=ro', uri=True, timeout=5)
    else:
        connection = sqlite3.connect(db_path, timeout=5)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.row_factory = sqlite3.Row
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_task_tables(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with connect(db_path) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY NOT NULL,
                user_id TEXT NOT NULL CHECK (length(trim(user_id)) > 0),
                description TEXT NOT NULL CHECK (length(trim(description)) > 0),
                status TEXT NOT NULL DEFAULT 'ACTIVE'
                    CHECK (status IN ('ACTIVE', 'COMPLETED', 'FAILED', 'CANCELLED')),
                period INTEGER CHECK (period IS NULL OR
                    (typeof(period) = 'integer' AND period > 0)),
                created_at TEXT NOT NULL,
                last_executed_at TEXT
            )"""
        )
        _allow_cancelled_tasks(connection)
        connection.execute(
            """CREATE TABLE IF NOT EXISTS actions (
                action_id TEXT PRIMARY KEY NOT NULL,
                task_id TEXT REFERENCES tasks(task_id),
                exec_at TEXT NOT NULL,
                description TEXT NOT NULL CHECK (length(trim(description)) > 0),
                status TEXT NOT NULL DEFAULT 'PENDING'
                    CHECK (status IN ('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED')),
                attempt_count INTEGER NOT NULL DEFAULT 0
                    CHECK (typeof(attempt_count) = 'integer' AND attempt_count >= 0),
                created_at TEXT NOT NULL
            )"""
        )
        columns = {row[1] for row in connection.execute('PRAGMA table_info(actions)')}
        renamed = 'instructions' in columns and 'description' not in columns
        if renamed:
            connection.execute('ALTER TABLE actions RENAME COLUMN instructions TO description')
        if 'context' not in columns:
            connection.execute("ALTER TABLE actions ADD COLUMN context TEXT NOT NULL DEFAULT '{}'")
        if renamed:
            for row in connection.execute('SELECT action_id,description,context FROM actions').fetchall():
                facts = json.loads(row['context'])
                facts.setdefault('legacy_details',row['description'])
                connection.execute('UPDATE actions SET context=? WHERE action_id=?',(json.dumps(facts),row['action_id']))
        # Preserve legacy timing and message facts before removing shortcut fields.
        if 'scheduled_for' in columns or 'notification' in columns:
            for row in connection.execute('SELECT * FROM actions').fetchall():
                context = json.loads(row['context'])
                if 'scheduled_for' in columns and row['scheduled_for']:
                    context.setdefault('requested_time', row['scheduled_for'])
                if 'notification' in columns and row['notification']:
                    context.setdefault('legacy_message', row['notification'])
                connection.execute('UPDATE actions SET context=? WHERE action_id=?',
                                   (json.dumps(context), row['action_id']))
            for column in ('scheduled_for', 'notification'):
                if column in columns:
                    connection.execute('ALTER TABLE actions DROP COLUMN ' + column)
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_actions_task_id ON actions(task_id)"
        )
    _allow_taskless_actions(db_path)


def _allow_cancelled_tasks(connection: sqlite3.Connection) -> None:
    """Upgrade the original task CHECK constraint, retaining tasks and FK links."""
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    ).fetchone()[0]
    if "'CANCELLED'" in schema:
        return
    # SQLite CHECK constraints require a table rebuild. Disable FK enforcement
    # before the transaction, then verify every link before committing.
    connection.execute("PRAGMA foreign_keys = OFF")
    connection.execute("BEGIN IMMEDIATE")
    schema = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'tasks'"
    ).fetchone()[0]
    if "'CANCELLED'" not in schema:
        definition = schema[schema.index("("):].replace("'FAILED'", "'FAILED', 'CANCELLED'", 1)
        connection.execute("CREATE TABLE tasks_upgrade " + definition)
        connection.execute("INSERT INTO tasks_upgrade SELECT * FROM tasks")
        connection.execute("DROP TABLE tasks")
        connection.execute("ALTER TABLE tasks_upgrade RENAME TO tasks")
        if connection.execute("PRAGMA foreign_key_check").fetchall():
            raise sqlite3.IntegrityError("Task migration would break foreign key links")
    connection.commit()
    connection.execute("PRAGMA foreign_keys = ON")


def _allow_taskless_actions(db_path: Path) -> None:
    """Retain action IDs and dependent leases/outbox rows during the upgrade."""
    with connect(db_path) as db:
        columns = {row['name']: row for row in db.execute('PRAGMA table_info(actions)')}
        if 'user_id' not in columns:
            db.execute('ALTER TABLE actions ADD COLUMN user_id TEXT')
            db.execute('UPDATE actions SET user_id=(SELECT user_id FROM tasks WHERE tasks.task_id=actions.task_id) WHERE user_id IS NULL AND task_id IS NOT NULL')
    with connect(db_path) as db:
        if not next(row for row in db.execute('PRAGMA table_info(actions)') if row['name']=='task_id')['notnull']:
            return
        db.execute('PRAGMA foreign_keys=OFF')
        db.execute('BEGIN IMMEDIATE')
        schema = db.execute("SELECT sql FROM sqlite_master WHERE name='actions'").fetchone()[0]
        indexes = [row[0] for row in db.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='actions' AND sql IS NOT NULL")]
        definition = schema[schema.index('('):].replace('task_id TEXT NOT NULL', 'task_id TEXT')
        db.execute('CREATE TABLE actions_upgrade ' + definition)
        db.execute('INSERT INTO actions_upgrade SELECT * FROM actions')
        db.execute('DROP TABLE actions')
        db.execute('ALTER TABLE actions_upgrade RENAME TO actions')
        for index in indexes:
            db.execute(index)
        if db.execute('PRAGMA foreign_key_check').fetchall():
            raise ValueError('Taskless action migration failed foreign-key validation')
