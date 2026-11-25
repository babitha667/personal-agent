import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock

from legacy_contextualizer import Contextualizer
from agent_tools import AgentTools
from database import connect
from models import Message
from storage import MessageStore
from test_contextualizer import call, done, plan, TEST_TOOLS


class HistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'agent.db'
        self.store = MessageStore(self.path)
        self.water = Message.from_content('Remind me to drink water at 6 PM')
        self.sport = Message.from_content('Book a tennis lesson')
        self.private = Message.from_content('Private water instruction for another user')
        self.store.save(self.water, 'u1')
        self.store.save(self.sport, 'u1')
        self.store.save(self.private, 'u2')
        self.current = Message.from_content('Change water to 7 PM')
        self.store.save(self.current, 'u1')
        self.tools = AgentTools(self.path, 'u1', self.current)

    def test_search_returns_only_relevant_prior_owned_messages(self):
        result = self.tools.execute('find_messages', {'query': 'water', 'limit': 5})
        self.assertEqual([m['id'] for m in result['messages']], [self.water.id])
        self.assertIn('6 PM', result['messages'][0]['message'])

    def test_recent_messages_are_newest_first_and_exclude_current_and_later(self):
        self.store.save(Message.from_content('A later message'), 'u1')
        result = self.tools.execute('get_recent_messages', {'limit': 10})
        self.assertEqual([m['id'] for m in result['messages']], [self.sport.id, self.water.id])

    def test_search_ranks_more_word_matches_first(self):
        result = self.tools.execute('find_messages', {'query': 'water remind tennis', 'limit': 2})
        self.assertEqual(result['messages'][0]['id'], self.water.id)
        self.assertEqual(result['messages'][1]['id'], self.sport.id)

    def test_bad_arguments_and_identity_override_are_rejected(self):
        for name, arguments in [
            ('get_recent_messages', {'limit': 21}),
            ('get_recent_messages', {'limit': True}),
            ('get_recent_messages', {'limit': 0}),
            ('get_recent_messages', {'limit': 2, 'user_id': 'u2'}),
            ('find_messages', {'query': '', 'limit': 5}),
            ('find_messages', {'query': 3, 'limit': 5}),
        ]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.tools.execute(name, arguments)

    def test_long_messages_are_bounded_and_sql_values_are_not_executed(self):
        long_message = Message.from_content('water ' + 'x' * 3000)
        self.store.save(long_message, 'u1')
        current = Message.from_content('Find earlier water messages')
        self.store.save(current, 'u1')
        tools = AgentTools(self.path, 'u1', current)
        result = tools.execute('find_messages', {'query': 'water', 'limit': 1})
        self.assertEqual(len(result['messages'][0]['message']), 2000)
        self.assertTrue(result['messages'][0]['truncated'])
        tools.execute('find_messages', {'query': "water'; DROP TABLE messages; --", 'limit': 5})
        with connect(self.path) as connection:
            self.assertGreater(connection.execute('SELECT COUNT(*) FROM messages').fetchone()[0], 0)

    def test_full_history_reaches_qwen_without_search(self):
        model = Mock()
        model.chat.return_value = call('apply_plan', plan())
        result = Contextualizer(self.store, llm=model, tool_definitions=TEST_TOOLS).receive_message(Message.from_content('Make the water reminder daily'))
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(model.chat.call_count, 1)
        prompt = json.dumps(model.chat.call_args.args[0])
        self.assertIn(self.water.message, prompt)
        self.assertNotIn(self.private.message, prompt)
        names = {tool['function']['name'] for tool in model.chat.call_args.kwargs['tools']}
        self.assertTrue({'create_reminder','ask_reminder_time'} <= names)
        self.assertFalse({'find_messages','find_tasks','get_task','get_recent_messages'} & names)
        self.assertNotIn('read tools', prompt)

    def test_existing_messages_are_preserved_and_assigned_to_u1(self):
        legacy_path = self.path.parent / 'legacy.db'
        with connect(legacy_path) as connection:
            connection.execute('CREATE TABLE messages (id TEXT PRIMARY KEY, timestamp TEXT, message TEXT)')
            connection.execute("INSERT INTO messages VALUES ('old', '2026-10-24', 'original text')")
        MessageStore(legacy_path)
        MessageStore(legacy_path)
        with connect(legacy_path) as connection:
            row = connection.execute('SELECT * FROM messages').fetchone()
            self.assertEqual(row['message'], 'original text')
            self.assertEqual(row['user_id'], 'u1')

    def test_hallucinated_search_calls_are_rejected(self):
        model = Mock()
        reads = call('find_tasks', {'query': 'water'})
        reads['tool_calls'] += call('find_messages', {'query': 'water', 'limit': 5})['tool_calls']
        model.chat.side_effect = [reads, call('apply_plan', plan())]
        result = Contextualizer(self.store, llm=model, tool_definitions=TEST_TOOLS).receive_message(
            Message.from_content('Make the water reminder daily')
        )
        self.assertEqual(model.chat.call_count, 2)
        self.assertEqual(len(result.tasks), 1)
        results = [item for item in model.chat.call_args.args[0] if item['role'] == 'tool']
        self.assertEqual([item['tool_name'] for item in results[:2]], ['find_tasks', 'find_messages'])
        self.assertIn('Call terminal tools alone', results[1]['content'])
