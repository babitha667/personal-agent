import sqlite3
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from database import connect, initialize_task_tables
from planning import save_plan, validate_plan


class PlanningTests(unittest.TestCase):
    def proposal(self, exec_at=None):
        return [{"description": "Drink water daily at 6 PM", "period": 86400,
                 "actions": [{"exec_at": exec_at, "description": "Assess water reminder for daily 6 PM Asia/Kolkata; reschedule if needed"}]}]

    def test_immediate_action_is_due_now(self):
        before = datetime.now(timezone.utc)
        tasks, actions = validate_plan(self.proposal(), "u2")
        after = datetime.now(timezone.utc)
        execution_time = datetime.fromisoformat(actions[0].exec_at)
        self.assertLessEqual(before, execution_time)
        self.assertLessEqual(execution_time, after)
        self.assertEqual(actions[0].exec_at, actions[0].created_at)
        self.assertEqual(tasks[0].user_id, "u2")
        self.assertEqual(tasks[0].period, 86400)
        self.assertIn("6 PM", actions[0].description)

    def test_requested_time_is_preserved_and_normalized_to_utc(self):
        tasks, actions = validate_plan(self.proposal("2099-10-24T18:00:00+05:30"), "u1")
        self.assertEqual(actions[0].exec_at, "2099-10-24T12:30:00+00:00")
        self.assertNotEqual(actions[0].exec_at, actions[0].created_at)

    def test_invalid_naive_and_past_times_are_rejected(self):
        for value in ("invalid", "2099-10-24T18:00:00", "2000-01-01T00:00:00+00:00"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_plan(self.proposal(value), "u1")

    def test_failed_action_insert_rolls_back_task_insert(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            initialize_task_tables(path)
            tasks, actions = validate_plan(self.proposal(), "u1")
            invalid_action = replace(actions[0], task_id="missing-task")
            with self.assertRaises(sqlite3.IntegrityError):
                save_plan(path, tasks, [invalid_action])
            with connect(path) as connection:
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)
                self.assertEqual(connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0], 0)
