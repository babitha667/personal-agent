import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta, timezone
from threading import Event
from time import monotonic, sleep

from legacy_contextualizer import Contextualizer
from action_queue import ActionQueue
from database import connect
from llm import LLMError, ModelTimeoutError
from message_queue import MessageQueue, MessageWorker
from models import Message
from storage import MessageStore
from test_contextualizer import call, plan, TEST_TOOLS


class MessageQueueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'agent.db'
        self.store = MessageStore(self.path)
        self.model = Mock()
        self.contextualizer = Contextualizer(self.store, llm=self.model, tool_definitions=TEST_TOOLS)
        self.queue = MessageQueue(self.path, 'u1')
        self.output = []
        self.worker = MessageWorker(self.queue, self.contextualizer)

    def enqueue(self, text='Hello'):
        message = Message.from_content(text)
        self.queue.enqueue(message)
        return message

    def test_fifo_queue_accepts_messages_while_planning_and_keeps_conversation_order(self):
        first = self.enqueue('First')
        second = self.enqueue('Second')
        def infer(messages, **kwargs):
            if messages[-1]['content'].split('):\n',1)[-1] == 'First':
                self.enqueue('Third while busy')
            return call('respond_chat', {'reply': 'Answer to '+messages[-1]['content'].split('):\n',1)[-1]})
        self.model.chat.side_effect = infer
        for _ in range(3):
            self.assertTrue(self.worker.run_once())
        self.assertFalse(self.worker.run_once())
        self.assertEqual(self.queue.status(first.id), 'DONE')
        self.assertEqual(self.queue.status(second.id), 'DONE')
        with connect(self.path) as db:
            texts = [r[0] for r in db.execute('SELECT message FROM messages ORDER BY rowid')]
        self.assertEqual(texts, ['First','Answer to First','Second','Answer to Second','Third while busy','Answer to Third while busy'])

    def test_background_planner_does_not_block_new_submissions(self):
        started, release, finished = Event(), Event(), Event()
        def infer(messages, **kwargs):
            if messages[-1]['content'].split('):\n',1)[-1] == 'First':
                started.set()
                release.wait(timeout=3)
            return call('respond_chat', {'reply': 'Done'})
        self.model.chat.side_effect = infer
        first = self.enqueue('First')
        output_count = []
        def notify(text):
            output_count.append(text)
            if len(output_count) == 2:
                finished.set()
        worker = MessageWorker(self.queue, self.contextualizer)
        worker.start()
        try:
            self.assertTrue(started.wait(timeout=3))
            second = self.enqueue('Second')
            self.assertEqual(self.queue.status(first.id), 'RUNNING')
            self.assertEqual(self.queue.status(second.id), 'PENDING')
            release.set()
            deadline=monotonic()+3
            while self.queue.status(second.id) != 'DONE' and monotonic() < deadline:
                sleep(0.01)
            self.assertEqual(self.queue.status(second.id), 'DONE')
        finally:
            release.set()
            worker.stop()

    def test_transient_retry_limit_retains_failed_message(self):
        message = self.enqueue()
        for attempt in range(3):
            _, token = self.queue.claim()
            retry = self.queue.fail(message.id, token, 'timeout', True)
            self.assertEqual(retry, attempt < 2)
            with connect(self.path) as db:
                db.execute("UPDATE message_inbox SET available_at='2000-01-01T00:00:00+00:00'")
        self.assertEqual(self.queue.status(message.id), 'FAILED')
        self.assertIsNone(self.queue.claim())

    def test_restart_and_other_user_isolation(self):
        message = self.enqueue()
        other = MessageQueue(self.path, 'u2')
        self.assertIsNone(other.claim())
        reopened = MessageQueue(self.path, 'u1')
        claimed, token = reopened.claim()
        self.assertEqual(claimed, message)
        self.assertIsNone(self.queue.claim())
        reopened.finish(message.id, token)
        self.assertEqual(self.queue.status(message.id), 'DONE')

    def test_timeout_retries_head_without_duplicate_input_or_overtaking(self):
        first = self.enqueue('First')
        second = self.enqueue('Second')
        self.model.chat.side_effect = [ModelTimeoutError('Ollama exceeded the 120s request budget. No plan was committed.'),
                                       call('respond_chat', {'reply':'Recovered'}), call('respond_chat', {'reply':'Next'})]
        with self.assertLogs('message_queue', level='ERROR'):
            self.worker.run_once()
        self.assertEqual(self.queue.status(first.id), 'PENDING')
        self.assertFalse(self.worker.run_once())
        with connect(self.path) as db:
            db.execute("UPDATE message_inbox SET available_at=?", ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),))
        self.worker.run_once()
        self.worker.run_once()
        with connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages WHERE id=?", (first.id,)).fetchone()[0],1)
        self.assertEqual(self.queue.status(second.id),'DONE')

    def test_expired_claim_is_recovered_and_old_worker_is_fenced(self):
        message = self.enqueue()
        _, old_token = self.queue.claim()
        with connect(self.path) as db:
            db.execute("UPDATE message_inbox SET lease_until='2000-01-01T00:00:00+00:00'")
        _, new_token = self.queue.claim()
        self.assertNotEqual(new_token,old_token)
        with self.assertRaisesRegex(ValueError,'lease expired'):
            self.contextualizer.receive_message(message,queue_token=old_token)
        self.model.chat.return_value=call('respond_chat',{'reply':'Recovered'})
        self.contextualizer.receive_message(message,queue_token=new_token)
        self.assertEqual(self.queue.status(message.id),'DONE')

    def test_task_commit_marks_inbox_done_even_if_reply_display_fails(self):
        message = self.enqueue('Make a reminder')
        self.model.chat.return_value=call('apply_plan',plan())
        with patch.object(self.contextualizer,'_save_reply',side_effect=RuntimeError('Simulated crash after commit')):
            with self.assertLogs('message_queue',level='ERROR'):
                self.worker.run_once()
        self.assertEqual(self.queue.status(message.id),'DONE')
        self.assertFalse(self.worker.run_once())
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
        self.assertTrue(any('was committed' in row['message'] for row in ActionQueue(self.path).pending_notifications()))

    def test_invalid_plan_failure_does_not_block_next_message(self):
        message=self.enqueue('Invalid work')
        self.enqueue('Hello')
        self.model.chat.side_effect=[LLMError('The model repeated a rejected plan.'),call('respond_chat',{'reply':'Hello'})]
        with self.assertLogs('message_queue',level='ERROR'):
            self.worker.run_once()
        self.assertEqual(self.queue.status(message.id),'FAILED')
        self.assertTrue(self.worker.run_once())

    def test_failure_notice_and_state_transition_commit_together(self):
        message=self.enqueue()
        _,token=self.queue.claim()
        with patch('outbox.enqueue_outgoing',side_effect=RuntimeError('Injected outbox failure')):
            with self.assertRaises(RuntimeError):
                self.queue.fail(message.id,token,'Failure',False)
        self.assertEqual(self.queue.status(message.id),'RUNNING')
        self.queue.fail(message.id,token,'Failure',False)
        self.assertEqual(self.queue.status(message.id),'FAILED')
        notices=ActionQueue(self.path).pending_notifications()
        self.assertEqual(len([r for r in notices if 'could not be processed' in r['message']]),1)

    def test_expired_poison_message_has_bounded_recovery(self):
        message=self.enqueue()
        self.queue.claim()
        with connect(self.path) as db:
            db.execute("UPDATE message_inbox SET attempts=3,lease_until='2000-01-01T00:00:00+00:00'")
        self.assertIsNone(self.queue.claim())
        self.assertEqual(self.queue.status(message.id),'FAILED')

    def test_stale_worker_does_not_emit_a_failure_or_revoke_current_claim(self):
        from message_queue import StaleMessageClaim
        message=self.enqueue()
        self.contextualizer.receive_message=Mock(side_effect=StaleMessageClaim('Reclaimed'))
        self.worker.run_once()
        self.assertEqual(self.queue.status(message.id),'RUNNING')
        self.assertEqual([r['message'] for r in ActionQueue(self.path).pending_notifications()],['👍 ACK'])
