"""SQLite storage for incoming messages."""

import sqlite3
import re
from pathlib import Path
from typing import ContextManager

from database import connect
from models import Message


class MessageStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # Upgrade databases created before the column names were simplified.
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(messages)")
            }
            for old, new in (
                ("messageId", "id"),
                ("MessageTimestamp", "timestamp"),
                ("MessageContent", "message"),
                ("content", "message"),
            ):
                if old in columns:
                    connection.execute(
                        f'ALTER TABLE messages RENAME COLUMN "{old}" TO "{new}"'
                    )
            connection.execute(
                """CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY NOT NULL,
                    timestamp TEXT NOT NULL,
                    message TEXT NOT NULL
                        CHECK (length(trim(message)) > 0),
                    user_id TEXT NOT NULL DEFAULT 'u1'
                        CHECK (length(trim(user_id)) > 0)
                )"""
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(messages)")}
            if "user_id" not in columns:
                connection.execute(
                    "ALTER TABLE messages ADD COLUMN user_id TEXT NOT NULL DEFAULT 'u1' "
                    "CHECK (length(trim(user_id)) > 0)"
                )
            if "role" not in columns:
                connection.execute("ALTER TABLE messages ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
            if "kind" not in columns:
                connection.execute("ALTER TABLE messages ADD COLUMN kind TEXT NOT NULL DEFAULT 'chat'")
            if "pending_activity" not in columns:
                connection.execute('ALTER TABLE messages ADD COLUMN pending_activity TEXT')
            connection.execute("CREATE INDEX IF NOT EXISTS idx_messages_user_id ON messages(user_id)")

    def _connect(self) -> ContextManager[sqlite3.Connection]:
        return connect(self.db_path)

    def save(self, message: Message, user_id: str = "u1", *, role: str = "user", kind: str = "chat", queue_message_id: str | None = None, queue_token: str | None = None, outbox_id: str | None = None, pending_activity: str | None = None) -> None:
        if role not in {'user', 'assistant'} or kind not in {'chat', 'clarification', 'plan', 'error', 'notification'}:
            raise ValueError('Invalid conversation role or kind')
        # Bound parameters keep user content separate from SQL statements.
        with self._connect() as connection:
            if queue_message_id is not None:
                connection.execute("BEGIN IMMEDIATE")
                from message_queue import require_claim
                require_claim(connection, queue_message_id, queue_token)
            existing = connection.execute('SELECT timestamp,message,user_id,role,kind,pending_activity FROM messages WHERE id=?', (message.id,)).fetchone()
            if existing is not None:
                if tuple(existing) != (message.timestamp, message.message, user_id, role, kind, pending_activity):
                    raise ValueError('Message ID already has different content')
                return
            connection.execute(
                """INSERT INTO messages
                   (id, timestamp, message, user_id, role, kind, pending_activity)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (message.id, message.timestamp, message.message, user_id, role, kind, pending_activity),
            )
            if outbox_id is not None:
                from outbox import enqueue_outgoing
                enqueue_outgoing(connection, outbox_id, user_id, message.message, timestamp=message.timestamp)
            if queue_message_id is not None and role == 'assistant':
                connection.execute("UPDATE message_inbox SET status='DONE',token=NULL,lease_until=NULL WHERE message_id=? AND token=?",
                                   (queue_message_id, queue_token))

    def full_history(self, user_id: str, before_id: str) -> list[dict]:
        """All prior user and assistant turns, without limits or truncation."""
        with self._connect() as connection:
            return [dict(row) for row in connection.execute(
                "SELECT id, timestamp, message, role, kind, pending_activity FROM messages WHERE user_id=? AND rowid < "
                "(SELECT rowid FROM messages WHERE id=? AND user_id=?) ORDER BY rowid",
                (user_id, before_id, user_id),
            )]

    def history(self, user_id: str, before_id: str, limit: int,
                query: str | None = None) -> list[dict]:
        """Return bounded prior user messages; query uses ranked keyword matching."""
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("limit must be an integer between 1 and 20")
        tokens = []
        if query is not None:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be non-empty text; use get_recent_messages for recent context")
            tokens = list(dict.fromkeys(re.findall(r"\w+", query.lower())))[:20]
            if not tokens:
                raise ValueError("query must contain words")
        score = " + ".join(
            "CASE WHEN instr(lower(message), ?) > 0 THEN 1 ELSE 0 END" for _ in tokens
        ) or "0"
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT id, timestamp, substr(message, 1, 2000) AS message, "
                f"length(message) > 2000 AS truncated, ({score}) AS score FROM messages "
                "WHERE user_id = ? AND role = 'user' AND rowid < "
                "(SELECT rowid FROM messages WHERE id = ? AND user_id = ?) "
                + ("AND score > 0 " if tokens else "")
                + "ORDER BY score DESC, rowid DESC LIMIT ?",
                (*tokens, user_id, before_id, user_id, limit),
            ).fetchall()
        return [{"id": row["id"], "timestamp": row["timestamp"],
                 "message": row["message"], "truncated": bool(row["truncated"])} for row in rows]
