import json
import sqlite3
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

from legacy_contextualizer import Contextualizer
from action_store import ActionStore
from database import connect, initialize_task_tables
from llm import LLMError
from models import Message
from planning import save_plan, validate_plan
from storage import MessageStore
from task_store import TaskStore
from test_contextualizer import call, done, TEST_TOOLS
from agent_tools import AgentTools


def proposal(description="Water reminder daily at 6 PM"):
    return [{"description": description, "period": 86400,
             "actions": [{"exec_at": "2099-10-24T18:00:00+05:30",
                          "description": description}]}]


class TaskUpdateTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "agent.db"
        self.llm = Mock()
        self.contextualizer = Contextualizer(MessageStore(self.path), llm=self.llm, tool_definitions=TEST_TOOLS)
        self.old_tasks, self.old_actions = validate_plan(proposal(), "u1")
        save_plan(self.path, self.old_tasks, self.old_actions)
        self.old_id = self.old_tasks[0].task_id

    def output(self, tasks=None, cancellations=None, missing=None):
        self.llm.chat.side_effect = [
            call("find_tasks", {"query": ""}),
            done(missing) if missing else call("apply_plan", {
                "tasks": tasks or [], "cancel_task_ids": cancellations or [],
            }),
            LLMError("Stopped after rejected plan"),
        ]

    def test_preloaded_task_can_be_replaced_without_search_roundtrip(self):
        self.llm.chat.return_value = call('apply_plan', {
            'tasks': proposal('Water daily at 7 PM'), 'cancel_task_ids': [self.old_id]})
        result = self.contextualizer.receive_message(Message.from_content('Push my water nudge back an hour'))
        self.assertEqual(result.cancelled_task_ids, [self.old_id])
        self.assertEqual(self.llm.chat.call_count, 1)
        prompt = json.dumps(self.llm.chat.call_args.args[0])
        self.assertIn(self.old_id, prompt)
        self.assertIn(self.old_actions[0].exec_at, prompt)

    def test_all_user_records_are_preloaded_without_message_limits(self):
        statuses = ['COMPLETED', 'FAILED', 'CANCELLED']
        for status in statuses:
            tasks, actions = validate_plan(proposal('Historical ' + status), 'u1')
            save_plan(self.path, tasks, actions)
            with connect(self.path) as connection:
                connection.execute('UPDATE tasks SET status=? WHERE task_id=?', (status, tasks[0].task_id))
                connection.execute('UPDATE actions SET status=? WHERE action_id=?', (status, actions[0].action_id))
        long_text = 'FULL_MESSAGE_START ' + 'x' * 2500 + ' FULL_MESSAGE_END'
        self.contextualizer.message_store.save(Message.from_content(long_text), 'u1')
        for index in range(25):
            self.contextualizer.message_store.save(Message.from_content('History record ' + str(index)), 'u1')
        self.llm.chat.return_value = done()
        self.contextualizer.receive_message(Message.from_content('Hello'))
        prompt = json.dumps(self.llm.chat.call_args.args[0])
        self.assertIn(long_text, prompt)
        for index in range(25):
            self.assertIn('History record ' + str(index), prompt)
        for status in statuses:
            self.assertIn('Historical ' + status, prompt)

    def test_historical_tasks_in_context_cannot_be_cancelled(self):
        save_plan(self.path, [], [], [self.old_id], 'u1')
        self.llm.chat.side_effect = [call('apply_plan', {'tasks': [], 'cancel_task_ids': [self.old_id]}),
                                     done(['Which active task?'])]
        result = self.contextualizer.receive_message(Message.from_content('Cancel that again'))
        self.assertEqual(result.cancelled_task_ids, [])
        self.assertEqual(TaskStore(self.path).get(self.old_id).status, 'CANCELLED')
        self.assertIn('is CANCELLED', self.llm.chat.call_args.args[0][3]['content'])

    def test_preloaded_context_excludes_other_users(self):
        tasks, actions = validate_plan(proposal('PRIVATE_TASK'), 'u2')
        save_plan(self.path, tasks, actions)
        self.contextualizer.message_store.save(Message.from_content('PRIVATE_MESSAGE'), 'u2')
        self.contextualizer.message_store.save(Message.from_content('My earlier context'), 'u1')
        self.llm.chat.return_value = done()
        self.contextualizer.receive_message(Message.from_content('Hello'))
        prompt = json.dumps(self.llm.chat.call_args.args[0])
        self.assertNotIn('PRIVATE_TASK', prompt)
        self.assertNotIn('PRIVATE_MESSAGE', prompt)
        self.assertIn('My earlier context', prompt)

    def test_update_cancels_old_work_and_creates_replacement(self):
        self.output(proposal("Water reminder daily at 7 PM"), [self.old_id])
        result = self.contextualizer.receive_message(Message.from_content("Move my water reminder to 7 PM"))
        self.assertEqual(result.cancelled_task_ids, [self.old_id])
        self.assertEqual(TaskStore(self.path).get(self.old_id).status, "CANCELLED")
        self.assertEqual(ActionStore(self.path).get(self.old_actions[0].action_id).status, "CANCELLED")
        self.assertEqual(result.tasks[0].status, "ACTIVE")
        self.assertEqual(result.actions[0].status, "PENDING")
        self.assertNotEqual(result.tasks[0].task_id, self.old_id)
        prompt = self.llm.chat.call_args.args[0]
        read_result = next(item for item in prompt if item.get("tool_name") == "find_tasks")
        self.assertIn(self.old_id, prompt[0]["content"])
        self.assertIn("6 PM", prompt[0]["content"])
        self.assertTrue(prompt[1]["content"].endswith("Move my water reminder to 7 PM"))

    def test_pure_cancellation_preserves_completed_action_history(self):
        completed = replace(self.old_actions[0], action_id="completed", status="COMPLETED")
        ActionStore(self.path).create(completed)
        self.output(cancellations=[self.old_id])
        result = self.contextualizer.receive_message(Message.from_content("Cancel the water reminder"))
        self.assertEqual(result.tasks, [])
        self.assertEqual(TaskStore(self.path).get(self.old_id).status, "CANCELLED")
        self.assertEqual(ActionStore(self.path).get("completed").status, "COMPLETED")

    def test_get_task_returns_actions_only_for_owner(self):
        tools = AgentTools(self.path, "u1", Message.from_content("Change the reminder"))
        result = tools.execute("get_task", {"task_id": self.old_id})
        self.assertEqual(result["task"]["task_id"], self.old_id)
        self.assertEqual(result["task"]["actions"][0]["action_id"], self.old_actions[0].action_id)
        other_tools = AgentTools(self.path, "u2", Message.from_content("Read task"))
        with self.assertRaisesRegex(ValueError, "not found"):
            other_tools.execute("get_task", {"task_id": self.old_id})

    def test_model_can_read_details_then_replace_task(self):
        self.llm.chat.side_effect = [
            call("find_tasks", {"query": "water"}),
            call("get_task", {"task_id": self.old_id}),
            call("apply_plan", {"tasks": proposal("Water daily at 7 PM"),
                                "cancel_task_ids": [self.old_id]}),
        ]
        result = self.contextualizer.receive_message(Message.from_content("Move water reminder to 7 PM"))
        self.assertEqual(result.cancelled_task_ids, [self.old_id])
        messages = self.llm.chat.call_args.args[0]
        details = next(item for item in messages if item.get("tool_name") == "get_task")
        self.assertIn(self.old_actions[0].exec_at, messages[0]["content"])

    def test_another_users_tasks_are_hidden_and_cannot_be_cancelled(self):
        other_tasks, other_actions = validate_plan(proposal("Private other-user task"), "u2")
        save_plan(self.path, other_tasks, other_actions)
        other_id = other_tasks[0].task_id
        self.output(cancellations=[other_id])
        with self.assertRaises(LLMError):
            self.contextualizer.receive_message(Message.from_content("Cancel that task"))
        context = next(item["content"] for item in self.llm.chat.call_args.args[0] if item.get("tool_name") == "find_tasks")
        self.assertNotIn(other_id, context)
        self.assertNotIn("Private other-user task", context)
        self.assertEqual(TaskStore(self.path).get(other_id).status, "ACTIVE")
        with self.assertRaises(ValueError):
            save_plan(self.path, [], [], [other_id], "u1")

    def test_unclear_update_leaves_existing_work_untouched(self):
        self.output(missing=["Which reminder should change?"])
        result = self.contextualizer.receive_message(Message.from_content("Move it to 7"))
        self.assertEqual(result.tasks, [])
        self.assertEqual(TaskStore(self.path).get(self.old_id).status, "ACTIVE")
        self.assertEqual(ActionStore(self.path).get(self.old_actions[0].action_id).status, "PENDING")

    def test_duplicate_plan_can_leave_existing_task_unchanged(self):
        self.output()
        self.contextualizer.receive_message(Message.from_content("Remind me to drink water every day at 6 PM"))
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)

    def test_failed_replacement_rolls_back_cancellation(self):
        tasks, actions = validate_plan(proposal("Replacement"), "u1")
        broken_action = replace(actions[0], task_id="nonexistent")
        with self.assertRaises(sqlite3.IntegrityError):
            save_plan(self.path, tasks, [broken_action], [self.old_id], "u1")
        self.assertEqual(TaskStore(self.path).get(self.old_id).status, "ACTIVE")
        self.assertEqual(ActionStore(self.path).get(self.old_actions[0].action_id).status, "PENDING")
        self.assertIsNone(TaskStore(self.path).get(tasks[0].task_id))

    def test_stale_cancellation_is_rejected(self):
        save_plan(self.path, [], [], [self.old_id], "u1")
        tasks, actions = validate_plan(proposal("Replacement"), "u1")
        with self.assertRaises(ValueError):
            save_plan(self.path, tasks, actions, [self.old_id], "u1")
        self.assertIsNone(TaskStore(self.path).get(tasks[0].task_id))


class TaskMigrationTests(unittest.TestCase):
    def test_existing_tasks_and_action_links_survive_status_upgrade(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.db"
            # Reproduce the old CHECK constraint with an existing child action.
            with connect(path) as connection:
                connection.execute("""CREATE TABLE tasks (
                    task_id TEXT PRIMARY KEY NOT NULL, user_id TEXT NOT NULL,
                    description TEXT NOT NULL,
                    status TEXT CHECK (status IN ('ACTIVE', 'COMPLETED', 'FAILED')),
                    period INTEGER, created_at TEXT NOT NULL, last_executed_at TEXT)""")
            initialize_task_tables(path)
            tasks, actions = validate_plan(proposal(), "u1")
            save_plan(path, tasks, actions)
            # Use a second legacy fixture with children present during rebuild.
            with connect(path) as connection:
                connection.execute("PRAGMA foreign_keys = OFF")
                connection.execute("BEGIN IMMEDIATE")
                schema = connection.execute("SELECT sql FROM sqlite_master WHERE name='tasks'").fetchone()[0]
                definition = schema[schema.index('('):].replace(", 'CANCELLED'", "")
                connection.execute("CREATE TABLE legacy_tasks " + definition)
                connection.execute("INSERT INTO legacy_tasks SELECT * FROM tasks")
                connection.execute("DROP TABLE tasks")
                connection.execute("ALTER TABLE legacy_tasks RENAME TO tasks")
            initialize_task_tables(path)
            initialize_task_tables(path)
            self.assertEqual(TaskStore(path).get(tasks[0].task_id), tasks[0])
            self.assertEqual(ActionStore(path).get(actions[0].action_id), actions[0])
            save_plan(path, [], [], [tasks[0].task_id], "u1")
            with connect(path) as connection:
                self.assertEqual(connection.execute("PRAGMA foreign_key_check").fetchall(), [])
                self.assertEqual(connection.execute("PRAGMA foreign_keys").fetchone()[0], 1)
