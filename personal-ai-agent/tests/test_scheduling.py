"""Structured schedules and executor follow-ups are independent of English wording."""
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

from legacy_contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue, utc_now
from database import connect
from llm import LLMError
from models import Message
from planning import save_plan, validate_plan
from scheduling import resolve_execution_time
from storage import MessageStore
from test_contextualizer import call, TEST_TOOLS


class StructuredSchedulingTests(unittest.TestCase):
    def test_relative_arithmetic_and_validation(self):
        received = datetime(2026, 10, 25, tzinfo=timezone.utc)
        now = received + timedelta(seconds=20)
        self.assertEqual(resolve_execution_time({'delay_seconds': 10}, received, now), received + timedelta(seconds=10))
        self.assertEqual(resolve_execution_time(None, received, now), now)
        for invalid in ({'delay_seconds': True}, {'delay_seconds': -1}, {'delay_seconds': float('nan')},
                        {'delay_seconds': 1, 'extra': 2}, '2:00', '2026-10-25T10:00:00',
                        (received - timedelta(seconds=1)).isoformat()):
            with self.subTest(invalid=invalid), self.assertRaises((ValueError, OverflowError)):
                resolve_execution_time(invalid, received, now)

    def test_explicit_schedule_objects(self):
        received = datetime(2026,10,25,12,0,tzinfo=timezone.utc)
        now = received + timedelta(seconds=80)
        self.assertEqual(resolve_execution_time({'delay_seconds':300},received,now), received+timedelta(minutes=5))
        self.assertEqual(resolve_execution_time({'immediate':True},received,now),now)
        self.assertEqual(resolve_execution_time({'at':'2026-10-25T18:00:00+05:30'},received,now),
                         datetime(2026,10,25,12,30,tzinfo=timezone.utc))
        with self.assertRaises(ValueError):
            resolve_execution_time({'at':'2026-10-24T12:00:00+00:00'},received,now)
        with self.assertRaises(ValueError):
            resolve_execution_time({'immediate':False},received,now)

    def test_model_schema_requires_explicit_schedule_choice(self):
        from agent_tools import ACTION_SCHEMA
        branches=ACTION_SCHEMA['properties']['exec_at']['anyOf']
        self.assertTrue(all(branch['type']=='object' for branch in branches))
        self.assertEqual([branch['required'] for branch in branches],
                         [['delay_seconds'], ['at'], ['immediate']])

    def test_existing_database_migration_preserves_actions(self):
        from database import initialize_task_tables
        from action_store import ActionStore
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.db'
            initialize_task_tables(path)
            tasks, actions = validate_plan([{'description': 'Existing work', 'period': None,
                'actions': [{'exec_at': None, 'description': 'Existing description'}]}], 'u1')
            save_plan(path, tasks, actions)
            with connect(path) as connection:
                connection.execute('ALTER TABLE actions ADD COLUMN notification TEXT')
            initialize_task_tables(path)
            initialize_task_tables(path)
            self.assertEqual(ActionStore(path).get(actions[0].action_id), actions[0])
            with connect(path) as connection:
                self.assertEqual(connection.execute('PRAGMA foreign_key_check').fetchall(), [])

    def test_overdue_followup_is_executed_by_model_after_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.db'
            model = Mock()
            model.chat.return_value = call('apply_plan', {'tasks': [{
                'description': 'Write plan', 'period': None,
                'actions': [{'description': 'Notify user', 'exec_at': {'delay_seconds': 10},
                             'notification': 'Time to write your plan.'}],
            }], 'cancel_task_ids': []})
            received = utc_now() - timedelta(seconds=17)
            result = Contextualizer(MessageStore(path), llm=model, tool_definitions=TEST_TOOLS).receive_message(
                Message('m1', received.isoformat(), 'Give me a nudge about my plan shortly'))
            self.assertEqual(result.actions[0].context['requested_time'], (received + timedelta(seconds=10)).isoformat())
            queue = ActionQueue(path)
            claim = queue.claim_due()
            from mock_execution import executor_model as scripted_model
            executor_model = scripted_model()
            ActionExecutor(queue, llm=executor_model).execute(claim)
            self.assertEqual(executor_model.chat.call_count,1)
            self.assertIsNone(queue.claim_due())
            with connect(path) as connection:
                self.assertEqual(connection.execute('SELECT status FROM actions WHERE action_id=?', (claim.action.action_id,)).fetchone()[0], 'COMPLETED')
                self.assertEqual(connection.execute('SELECT message FROM notifications WHERE message=?', ('Time to write your plan.',)).fetchone()[0], 'Time to write your plan.')
            self.assertIsNone(result.reply)
            self.assertEqual(model.chat.call_count, 1)

    def test_recurring_action_keeps_context(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'agent.db'
            MessageStore(path)
            from database import initialize_task_tables
            initialize_task_tables(path)
            tasks, actions = validate_plan([{'description': 'Stretch', 'period': 60,
                'actions': [{'exec_at': None, 'description': 'Notify', 'notification': 'Stretch!'}]}], 'u1')
            save_plan(path, tasks, actions)
            queue = ActionQueue(path)
            from mock_execution import executor_model
            ActionExecutor(queue, llm=executor_model()).execute(queue.claim_due())
            with connect(path) as connection:
                successor = connection.execute("SELECT context FROM actions WHERE status='PENDING'").fetchone()
                self.assertIsNone(successor)
                self.assertEqual(connection.execute('SELECT status FROM tasks').fetchone()[0],'ACTIVE')

    def test_keywords_do_not_force_creation_for_explanations_or_negations(self):
        for text in ('Explain how to remind me to drink water in 30s', "Don't remind me to drink water in 30s"):
            with self.subTest(text=text), tempfile.TemporaryDirectory() as directory:
                model = Mock()
                model.chat.return_value = call('respond_chat', {'reply': 'Understood.'})
                result = Contextualizer(MessageStore(Path(directory)/'agent.db'), llm=model, tool_definitions=TEST_TOOLS).receive_message(Message.from_content(text))
                self.assertEqual(result.tasks, [])
                self.assertEqual(model.chat.call_count, 1)

    def test_native_text_is_not_an_executable_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Mock()
            model.chat.return_value = {'role': 'assistant', 'content': '{"reply":"Saved!","missing_information":[]}'}
            contextualizer = Contextualizer(MessageStore(Path(directory)/'agent.db'), llm=model, tool_definitions=TEST_TOOLS)
            result = contextualizer.receive_message(Message.from_content('Nudge me in half a minute'))
            self.assertEqual(result.tasks, [])
