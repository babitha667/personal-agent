import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from legacy_contextualizer import Contextualizer
from database import connect
from llm import LLMError
from models import Message
from planning import save_plan, validate_plan
from storage import MessageStore
from test_contextualizer import call, TEST_TOOLS


class ConversationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'agent.db'
        self.store = MessageStore(self.path)
        self.model = Mock()
        self.contextualizer = Contextualizer(self.store, llm=self.model, tool_definitions=TEST_TOOLS)

    def test_short_answer_sees_saved_assistant_question_after_restart(self):
        question = 'Which electrolytes do you prefer?'
        self.model.chat.return_value = call('ask_clarification', {'reply': question, 'missing_information': ['Electrolyte preference']})
        self.contextualizer.receive_message(Message.from_content('I like electrolytes in water'))
        other_model = Mock()
        other_model.chat.return_value = call('respond_chat', {'reply': 'Understood, sodium.'})
        Contextualizer(MessageStore(self.path), llm=other_model, tool_definitions=TEST_TOOLS).receive_message(Message.from_content('Sodium'))
        prompt = json.dumps(other_model.chat.call_args.args[0])
        self.assertIn(question, prompt)
        self.assertIn('"role": "assistant"', prompt)
        with connect(self.path) as db:
            self.assertEqual([row[0] for row in db.execute('SELECT role FROM messages ORDER BY rowid')],
                             ['user', 'assistant', 'user', 'assistant'])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0], 0)

    def test_confirmed_update_replaces_active_instructions_preserving_time(self):
        due = '2099-10-25T12:00:00+05:30'
        tasks, actions = validate_plan([{'description': 'Drink water', 'period': None,
            'actions': [{'exec_at': due, 'description': 'Notify about water', 'notification': 'Drink water!'}]}], 'u1')
        save_plan(self.path, tasks, actions)
        self.model.chat.return_value = call('ask_clarification', {
            'reply': 'Should I add your sodium preference to the pending water reminder?',
            'missing_information': ['Confirm reminder update']})
        self.contextualizer.receive_message(Message.from_content('I like sodium in my water'))
        self.model.chat.reset_mock()
        self.model.chat.return_value = call('apply_plan', {'cancel_task_ids': [tasks[0].task_id], 'tasks': [{
            'description': 'Drink water with preferred electrolytes', 'period': None,
            'actions': [{'exec_at': due, 'description': 'Notify user to drink water with their sodium preference',
                         'notification': 'Time for your water with electrolytes.'}]}]})
        result = self.contextualizer.receive_message(Message.from_content('yes'))
        self.assertIn('sodium', result.actions[0].description)
        self.assertEqual(result.actions[0].context.get('requested_time'), actions[0].exec_at)
        self.assertEqual(result.cancelled_task_ids, [tasks[0].task_id])
        self.assertEqual(self.model.chat.call_count, 1)
        self.assertIn('Should I add your sodium preference', json.dumps(self.model.chat.call_args.args[0]))
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT status FROM actions WHERE action_id=?', (actions[0].action_id,)).fetchone()[0], 'CANCELLED')

    def test_completed_task_cannot_be_cancelled_and_new_request_can_recover(self):
        tasks, actions = validate_plan([{'description': 'Old reminder', 'period': None,
            'actions': [{'exec_at': None, 'description': 'Notify'}]}], 'u1')
        save_plan(self.path, tasks, actions)
        with connect(self.path) as db:
            db.execute("UPDATE tasks SET status='COMPLETED'")
            db.execute("UPDATE actions SET status='COMPLETED'")
        proposal = {'description': 'New reminder', 'period': None,
                    'actions': [{'exec_at': {'delay_seconds': 180}, 'description': 'Notify', 'notification': 'Water!'}]}
        self.model.chat.side_effect = [call('apply_plan', {'tasks': [proposal], 'cancel_task_ids': [tasks[0].task_id]}),
                                       call('apply_plan', {'tasks': [proposal], 'cancel_task_ids': []})]
        result = self.contextualizer.receive_message(Message.from_content('Give me a new reminder in three minutes'))
        self.assertEqual(len(result.actions), 1)
        messages = self.model.chat.call_args.args[0]
        self.assertIn('is COMPLETED', messages[3]['content'])
        names = {t['function']['name'] for t in self.model.chat.call_args.kwargs['tools']}
        self.assertIn('create_reminder', names)

    def test_placeholder_zero_delay_recovers_by_asking_time_without_writes(self):
        self.model.chat.side_effect = [call('apply_plan', {'tasks': [{
            'description': 'Drink water', 'period': None, 'actions': [{
                'description': 'Notify', 'exec_at': {'delay_seconds': 0}, 'notification': 'Water!'}]}],
            'cancel_task_ids': []}), call('ask_clarification', {
                'reply': 'When would you like me to remind you?', 'missing_information': ['Reminder time']})]
        result = self.contextualizer.receive_message(Message.from_content('Can you set a reminder to drink water?'))
        self.assertEqual(result.reply, 'When would you like me to remind you?')
        self.assertEqual([t['function']['name'] for t in self.model.chat.call_args.kwargs['tools']], ['ask_reminder_time', 'ask_clarification'])
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0], 0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM actions').fetchone()[0], 0)
        self.model.chat.side_effect = None
        self.model.chat.return_value = call('apply_plan', {'tasks': [{
            'description': 'Drink water', 'period': None, 'actions': [{
                'description': 'Notify', 'exec_at': {'delay_seconds': 300}, 'notification': 'Water!'}]}],
            'cancel_task_ids': []})
        followup = self.contextualizer.receive_message(Message.from_content('In five minutes'))
        self.assertEqual(len(followup.actions), 1)
        self.assertIn('When would you like me to remind you?', json.dumps(self.model.chat.call_args.args[0]))

    def test_identical_rejected_plan_stops_without_three_slow_retries(self):
        invalid = {'tasks': [], 'cancel_task_ids': ['invented']}
        self.model.chat.return_value = call('apply_plan', invalid)
        with self.assertRaisesRegex(LLMError, 'repeated a rejected plan'):
            self.contextualizer.receive_message(Message.from_content('Change it'))
        self.assertEqual(self.model.chat.call_count, 2)
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0], 0)
