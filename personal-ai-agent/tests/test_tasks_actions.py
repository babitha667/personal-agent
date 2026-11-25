import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from action_store import ActionStore
from database import connect
from models import Action, Message, Task
from storage import MessageStore
from task_store import TaskStore


class TaskActionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "agent.db"
        self.messages = MessageStore(self.path)
        self.messages.save(Message.from_content("Keep this message"))
        self.tasks = TaskStore(self.path)
        self.actions = ActionStore(self.path)
        self.task = Task(
            task_id="t123", user_id="u1",
            description="Remind me to drink water every day at 6 PM",
            status="ACTIVE", period=86400,
            created_at="2026-10-24T18:00:00+05:30",
        )
        self.action = Action(
            action_id="a123", task_id="t123",
            exec_at="2026-10-24T18:00:00+05:30",
            description="Send a reminder to drink water",
            status="PENDING", attempt_count=0,
            created_at="2026-10-24T17:00:00+05:30",
        )

    def test_round_trip_and_restart_preserve_all_tables(self):
        self.tasks.create(self.task)
        self.actions.create(self.action)
        self.assertEqual(TaskStore(self.path).get("t123"), self.task)
        self.assertEqual(ActionStore(self.path).get("a123"), self.action)
        self.assertIsNone(self.tasks.get("missing"))
        self.assertIsNone(self.actions.get("missing"))
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 1)

    def test_one_time_task_and_last_execution(self):
        task = replace(self.task, period=None, last_executed_at=self.action.exec_at)
        self.tasks.create(task)
        self.assertEqual(self.tasks.get("t123"), task)

    def test_action_requires_existing_task(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.actions.create(self.action)

    def test_invalid_status_period_attempts_and_duplicate_ids(self):
        for task in (replace(self.task, status="UNKNOWN"), replace(self.task, period=0)):
            with self.subTest(task=task), self.assertRaises(sqlite3.IntegrityError):
                self.tasks.create(task)
        self.tasks.create(self.task)
        with self.assertRaises(sqlite3.IntegrityError):
            self.tasks.create(self.task)
        for action in (replace(self.action, status="UNKNOWN"), replace(self.action, attempt_count=-1)):
            with self.subTest(action=action), self.assertRaises(sqlite3.IntegrityError):
                self.actions.create(action)


if __name__ == "__main__":
    unittest.main()
