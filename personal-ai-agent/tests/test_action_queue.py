import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

from action_queue import ActionQueue, StaleClaim, utc_now
from action_store import ActionStore
from database import connect
from models import Action, Task
from planning import save_plan
from task_store import TaskStore


class QueueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'agent.db'
        self.queue = ActionQueue(self.path)
        self.now = utc_now()
        self.task = Task('t1', 'u1', 'Water reminder', 'ACTIVE', None, self.now.isoformat())
        self.action = Action('a1', 't1', (self.now - timedelta(seconds=1)).isoformat(),
                             'Remind user to drink water', 'PENDING', 0, self.now.isoformat())
        save_plan(self.path, [self.task], [self.action])

    def test_only_due_owned_active_actions_are_claimed_in_time_order(self):
        future = replace(self.action, action_id='future', exec_at=(self.now + timedelta(days=1)).isoformat())
        ActionStore(self.path).create(future)
        self.assertIsNone(ActionQueue(self.path, 'u2').claim_due(self.now))
        claim = self.queue.claim_due(self.now)
        self.assertEqual(claim.action.action_id, 'a1')
        self.assertEqual(claim.action.attempt_count, 1)
        self.assertIsNone(self.queue.claim_due(self.now))

    def test_two_workers_cannot_claim_same_action(self):
        other = ActionQueue(self.path)
        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda q: q.claim_due(self.now), [self.queue, other]))
        self.assertEqual(sum(result is not None for result in results), 1)

    def test_lease_recovers_after_restart_and_old_worker_is_fenced(self):
        first = self.queue.claim_due(self.now)
        later = self.now + timedelta(seconds=601)
        second = ActionQueue(self.path).claim_due(later)
        self.assertEqual(second.action.attempt_count, 2)
        with self.assertRaises(StaleClaim):
            self.queue.finish(first, 'completed', [], 'stale output', now=later)
        self.queue.finish(second, 'completed', [], 'valid reminder', now=later)
        self.assertEqual(len(self.queue.pending_notifications()), 1)
        with self.assertRaises(StaleClaim):
            self.queue.finish(second, 'completed', [], 'duplicate', now=later)

    def test_retry_backoff_and_failure_limit(self):
        now = self.now
        for attempt in range(1, 4):
            claim = self.queue.claim_due(now)
            self.assertEqual(claim.action.attempt_count, attempt)
            self.queue.retry(claim, now)
            if attempt < 3:
                self.assertIsNone(self.queue.claim_due(now))
                now += timedelta(seconds=5 * 2 ** (attempt - 1))
        self.assertEqual(ActionStore(self.path).get('a1').status, 'FAILED')
        self.assertEqual(TaskStore(self.path).get('t1').status, 'FAILED')

    def test_cancelled_task_cannot_commit_or_be_revived(self):
        claim = self.queue.claim_due(self.now)
        save_plan(self.path, [], [], ['t1'], 'u1')
        with self.assertRaises(StaleClaim):
            self.queue.finish(claim, 'completed', [], 'do not deliver', now=self.now)
        self.queue.retry(claim, self.now)
        self.assertEqual(ActionStore(self.path).get('a1').status, 'CANCELLED')
        self.assertEqual(self.queue.pending_notifications(), [])

    def test_followup_is_durable_and_keeps_task_active(self):
        claim = self.queue.claim_due(self.now)
        later = self.now + timedelta(hours=1)
        ids = self.queue.finish(claim, 'completed', [{'exec_at': later.isoformat(), 'description': 'Check progress'}],
                                'Follow-up queued', now=self.now)
        self.assertEqual(len(ids), 1)
        self.assertEqual(TaskStore(self.path).get('t1').status, 'ACTIVE')
        self.assertIsNone(ActionQueue(self.path).claim_due(self.now))
        next_claim = ActionQueue(self.path).claim_due(later)
        self.assertEqual(next_claim.action.action_id, ids[0])
        self.queue.finish(next_claim, 'skipped', [], None, now=later)
        self.assertEqual(TaskStore(self.path).get('t1').status, 'COMPLETED')

    def test_queue_leaves_recurrence_to_continuous_controller(self):
        with connect(self.path) as connection:
            connection.execute('UPDATE tasks SET period=86400 WHERE task_id=?', ('t1',))
        claim=self.queue.claim_due(self.now)
        ids=self.queue.finish(claim,'completed',[],'Drink water',now=self.now)
        self.assertEqual(ids,[])
        task=TaskStore(self.path).get('t1')
        self.assertEqual(task.status,'ACTIVE')
        self.assertEqual(task.last_executed_at,self.now.isoformat())

    def test_expired_final_attempt_marks_task_failed(self):
        with connect(self.path) as connection:
            connection.execute('UPDATE actions SET attempt_count=2 WHERE action_id=?', ('a1',))
        self.queue.claim_due(self.now)
        self.assertIsNone(self.queue.claim_due(self.now + timedelta(seconds=601)))
        self.assertEqual(ActionStore(self.path).get('a1').status, 'FAILED')
        self.assertEqual(TaskStore(self.path).get('t1').status, 'FAILED')
