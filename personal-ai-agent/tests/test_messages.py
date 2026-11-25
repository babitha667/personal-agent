import sqlite3
from contextlib import closing
import tempfile
import unittest
from unittest.mock import Mock
from datetime import datetime
from pathlib import Path
from uuid import UUID

from legacy_contextualizer import Contextualizer
from models import Message
from storage import MessageStore
from test_contextualizer import done, TEST_TOOLS


class MessageTests(unittest.TestCase):
    def test_legacy_columns_are_renamed_without_losing_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            with closing(sqlite3.connect(path)) as connection, connection:
                connection.execute(
                    "CREATE TABLE messages (messageId TEXT PRIMARY KEY NOT NULL, "
                    "MessageTimestamp TEXT NOT NULL, MessageContent TEXT NOT NULL)"
                )
                connection.execute(
                    "INSERT INTO messages VALUES (?, ?, ?)",
                    ("old-id", "2026-10-24T12:00:00+00:00", "Saved message"),
                )
            MessageStore(path)
            store = MessageStore(path)
            store.save(Message.from_content("New message"))
            with closing(sqlite3.connect(path)) as connection, connection:
                columns = [row[1] for row in connection.execute("PRAGMA table_info(messages)")]
                rows = connection.execute("SELECT id, message FROM messages ORDER BY rowid").fetchall()
            self.assertEqual(columns, ["id", "timestamp", "message", "user_id", "role", "kind", "pending_activity"])
            self.assertEqual(rows[0], ("old-id", "Saved message"))
            self.assertEqual(rows[1][1], "New message")

    def test_message_has_unique_id_and_utc_timestamp(self):
        first = Message.from_content("Hello")
        second = Message.from_content("Hello")
        UUID(first.id)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(datetime.fromisoformat(first.timestamp).utcoffset().total_seconds(), 0)
        with self.assertRaises(ValueError):
            Message.from_content("  ")

    def test_messages_survive_reopening_store(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data" / "agent.db"
            first = Message.from_content("Let's meet; DROP TABLE messages; --")
            contextualizer = Contextualizer(MessageStore(path), tool_definitions=TEST_TOOLS, llm=Mock(chat=Mock(return_value=done() )))
            self.assertEqual(contextualizer.receive_message(first).incoming_message, first)
            second = Message.from_content("Hello again")
            reopened = Contextualizer(MessageStore(path), tool_definitions=TEST_TOOLS, llm=Mock(chat=Mock(return_value=done() )))
            self.assertEqual(reopened.receive_message(second).incoming_message, second)
            with closing(sqlite3.connect(path)) as connection, connection:
                rows = connection.execute(
                    "SELECT id, timestamp, message FROM messages WHERE role='user' ORDER BY rowid"
                ).fetchall()
            self.assertEqual(rows, [
                (first.id, first.timestamp, first.message),
                (second.id, second.timestamp, second.message),
            ])


if __name__ == "__main__":
    unittest.main()
