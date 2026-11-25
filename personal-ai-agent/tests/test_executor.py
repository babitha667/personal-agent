import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch
from threading import Event

from action_executor import ActionExecutor
from action_queue import ActionQueue, utc_now
from action_store import ActionStore
from controller import ActionController
from database import connect
from executor_tools import ExecutorTools
from llm import LLMError
from models import Action, Task
from planning import save_plan
from task_store import TaskStore
from test_contextualizer import call, TEST_TOOLS
from test_contextualizer import plan, TEST_TOOLS
from legacy_contextualizer import Contextualizer
import main as terminal


def finish(outcome='completed', followups=None):
    operations=[]
    if followups:
        operations.extend(call('add_actions',{'actions':followups})['tool_calls'])
    operations.extend(call('finish_action',{'outcome':'WAITING' if outcome=='needs_input' else outcome.upper(), 'reason':'Work processed', **({'task_status':'ACTIVE' if followups else 'COMPLETED'} if outcome=='completed' else {})})['tool_calls'])
    return {'role':'assistant','content':'','tool_calls':operations}


class ExecutorTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'agent.db'
        self.queue = ActionQueue(self.path)
        now = utc_now()
        self.task = Task('t1', 'u1', 'Remind user to drink water', 'ACTIVE', None, now.isoformat())
        self.action = Action('a1', 't1', (now - timedelta(seconds=1)).isoformat(),
                             'Notify user to drink water', 'PENDING', 0, now.isoformat())
        save_plan(self.path, [self.task], [self.action])
        self.model = Mock()
        self.notifications = []
        self.controller = ActionController(self.queue, ActionExecutor(self.queue, self.model), self.notifications.append)

    def test_due_notification_is_sent_through_native_executor(self):
        from test_generic_planner import batch
        self.model.chat.return_value=batch(('send_message', {'message':'Write your plan!'}),
            ('finish_action', {'outcome':'COMPLETED','reason':'Delivered'}))
        self.controller.run_once()
        self.assertEqual(self.notifications,['Write your plan!'])
        self.assertEqual(self.model.chat.call_count,1)

    def test_due_reminder_runs_tools_and_delivers_notification(self):
        self.model.chat.side_effect = [call('send_message', {'message': 'Time to drink water!'}), finish()]
        self.assertTrue(self.controller.run_once())
        self.assertEqual(self.notifications, ['Time to drink water!'])
        self.assertEqual(ActionStore(self.path).get('a1').status, 'COMPLETED')
        task = TaskStore(self.path).get('t1')
        self.assertEqual(task.status, 'COMPLETED')
        self.assertIsNotNone(task.last_executed_at)
        self.assertFalse(self.controller.run_once())
        self.assertEqual(self.model.chat.call_count, 2)

    def test_reminder_can_finish_with_notification_in_one_model_call(self):
        response = finish()
        response['tool_calls'].insert(0, call('send_message',{'message':'Time to drink water!'})['tool_calls'][0])
        self.model.chat.return_value = response
        self.controller.run_once()
        self.assertEqual(self.notifications, ['Time to drink water!'])
        self.assertEqual(self.model.chat.call_count, 1)
        self.assertEqual(TaskStore(self.path).get('t1').status, 'COMPLETED')
        self.assertIn('finish_action', [tool['function']['name'] for tool in self.model.chat.call_args.kwargs['tools']])



    def test_model_failure_discards_staged_effect_and_retries(self):
        self.model.chat.side_effect = [call('send_message', {'message': 'Not committed'}), LLMError('offline')]
        with self.assertLogs('controller', level='ERROR'):
            self.controller.run_once()
        self.assertEqual(self.notifications, [])
        self.assertEqual(self.queue.pending_notifications(), [])
        self.assertEqual(ActionStore(self.path).get('a1').status, 'PENDING')

    def test_reschedule_creates_future_action_without_waiting(self):
        self.model.chat.side_effect = [finish(followups=[{
            'exec_at': (utc_now()+timedelta(hours=1)).isoformat(), 'description': 'Notify user later',
        }])]
        self.controller.run_once()
        self.assertEqual(self.notifications, [])
        self.assertFalse(self.controller.run_once())
        self.assertEqual(TaskStore(self.path).get('t1').status, 'ACTIVE')

    def test_executor_cannot_run_task_planner_tools(self):
        tools = ExecutorTools(self.queue, self.queue.claim_due())
        with self.assertRaises(ValueError):
            tools.execute('apply_plan', {'tasks': [], 'cancel_task_ids': []})
        with self.assertRaises(ValueError):
            tools.execute('finish_action', {'outcome': 'completed', 'reason': 'Bad time',
                                          'follow_ups': [{'exec_at': '2000-01-01T00:00:00+00:00', 'description': 'Past'}]})

    def test_failed_delivery_is_retried_from_notification_outbox(self):
        self.model.chat.side_effect = [call('send_message', {'message': 'Durable notification'}), finish()]
        self.controller.notify = Mock(side_effect=RuntimeError('terminal unavailable'))
        with self.assertLogs('controller', level='ERROR'):
            self.controller.run_once()
        self.assertEqual(len(self.queue.pending_notifications()), 1)
        restarted = ActionController(ActionQueue(self.path), ActionExecutor(self.queue, self.model), self.notifications.append)
        restarted.deliver_notifications()
        self.assertEqual(self.notifications, ['Durable notification'])
        self.assertEqual(self.queue.pending_notifications(), [])

    def test_missing_details_keep_task_active_for_user_update(self):
        self.model.chat.side_effect = [call('send_message', {'message': 'When should I remind you?'}), finish('needs_input')]
        self.controller.run_once()
        self.assertEqual(TaskStore(self.path).get('t1').status, 'ACTIVE')
        self.assertEqual(ActionStore(self.path).get('a1').status, 'COMPLETED')


    def test_controller_keeps_polling_while_executor_is_busy(self):
        started, release, polled_while_busy = Event(), Event(), Event()
        extra = replace(self.action, action_id='a2')
        ActionStore(self.path).create(extra)

        def slow_execution(claim):
            started.set()
            release.wait(timeout=3)

        def observe_poll(message, *args):
            if 'controller_poll busy_workers' in message:
                polled_while_busy.set()

        executor = Mock()
        executor.execute.side_effect = slow_execution
        controller = ActionController(self.queue, executor, self.notifications.append,
                                      poll_seconds=0.01, max_workers=1)
        try:
            with patch('controller.logger.debug', side_effect=observe_poll):
                controller.start()
                self.assertTrue(started.wait(timeout=1))
                self.assertTrue(polled_while_busy.wait(timeout=1))
                self.assertEqual(executor.execute.call_count, 1)
                self.assertEqual(ActionStore(self.path).get('a2').status, 'PENDING')
        finally:
            controller.stop()
            release.set()
            for worker in controller.workers:
                worker.join(timeout=1)

    def test_terminal_planning_and_background_execution_end_to_end(self):
        root = self.path.parent / 'terminal_demo'
        planner = Mock()
        proposal = plan()
        proposal['tasks'][0]['period'] = None
        planner.chat.side_effect = [call('find_tasks', {'query': ''}), call('apply_plan', proposal)]
        executor_model = Mock()
        executor_model.chat.side_effect = [call('send_message', {'message': 'Time to drink water!'}), finish()]
        delivered = Event()
        inputs = iter(['Remind me to drink water now'])

        def read_input(prompt):
            try:
                return next(inputs)
            except StopIteration:
                self.assertTrue(delivered.wait(timeout=3), 'Background reminder did not arrive')
                raise EOFError

        def print_output(*args, **kwargs):
            if any('Time to drink water!' in str(arg) for arg in args):
                delivered.set()

        with patch.object(terminal, '__file__', str(root / 'main.py')), \
             patch.object(terminal, 'configure_logging', return_value=root / 'tmp' / 'agent.log'), \
             patch.object(terminal, 'Contextualizer', side_effect=lambda store, **kw: Contextualizer(store, llm=planner, tool_definitions=TEST_TOOLS, **kw)), \
             patch.object(terminal, 'ActionExecutor', side_effect=lambda queue, **kw: ActionExecutor(queue, llm=executor_model, **kw)), \
             patch.dict('os.environ', {'AGENT_USER_ID': 'u1', 'AGENT_WORKER_ENABLED': '1', 'AGENT_PROACTIVE_ENABLED':'0'}), \
             patch('builtins.input', side_effect=read_input), patch('builtins.print', side_effect=print_output):
            terminal.main()
        with connect(root / 'data' / 'agent.db') as connection:
            self.assertEqual(connection.execute('SELECT status FROM tasks').fetchone()[0], 'COMPLETED')
            self.assertEqual(connection.execute('SELECT status FROM actions').fetchone()[0], 'COMPLETED')

    def test_generic_tool_allowlist_and_read_actions_are_scoped(self):
        from executor_tools import EXECUTOR_TOOLS
        self.assertEqual({t['function']['name'] for t in EXECUTOR_TOOLS},
                         {'read_actions','add_actions','send_message','finish_action','send_email','read_email','draft_email','read_tasks','select_task','create_task','update_task','cancel_task','update_action','cancel_action'})
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        self.assertEqual(tools.execute('read_actions',{})['task']['task_id'],'t1')
        with self.assertRaises(ValueError): tools.execute('read_actions',{'task_id':'someone-else'})

    def test_immediate_followup_and_message_commit_atomically(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('add_actions',{'actions':[{'exec_at':{'immediate':True},'description':'Continue with the next step'}]})
        tools.execute('send_message',{'message':'I’m continuing with the next step.'})
        self.assertEqual(self.queue.pending_notifications(),[])
        result=tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Next step required'})
        self.assertEqual(len(result['follow_up_action_ids']),1)
        self.assertEqual(TaskStore(self.path).get('t1').status,'ACTIVE')
        self.assertIsNotNone(self.queue.claim_due())
        self.assertEqual(len(self.queue.pending_notifications()),1)

    def test_executor_cannot_generate_entire_workflow(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        with self.assertRaises(ValueError):
            tools.execute('add_actions',{'actions':[
                {'exec_at':{'immediate':True},'description':'First'},
                {'exec_at':{'delay_seconds':60},'description':'Second'}]})
        self.assertEqual(tools.follow_ups,[])

    def test_waiting_records_reason_and_keeps_task_active(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('send_message',{'message':'What time would you prefer?'})
        tools.execute('finish_action',{'outcome':'WAITING','reason':'Waiting for user to choose reminder time'})
        self.assertEqual(TaskStore(self.path).get('t1').status,'ACTIVE')
        with connect(self.path) as db:
            result=db.execute('SELECT * FROM action_results WHERE action_id=?',('a1',)).fetchone()
        self.assertEqual(result['outcome'],'WAITING')
        self.assertIn('choose reminder time',result['reason'])
        from continuous_user_processor import ContinuousUserProcessor
        processor=ContinuousUserProcessor(self.path);processor.register_user('u1')
        with connect(self.path) as db: db.execute('UPDATE tasks SET period=1')
        self.assertEqual(processor.trigger_recurring('u1',utc_now()+timedelta(hours=1)),0)

    def test_invalid_batch_does_not_publish_staged_success(self):
        invalid={'role':'assistant','content':'','tool_calls':[
            *call('send_message',{'message':'Incorrect success'})['tool_calls'],
            *call('add_actions',{'actions':[{'exec_at':'2000-01-01T00:00:00+00:00','description':'Invalid past action'}]})['tool_calls'],
            *finish()['tool_calls']]}
        self.model.chat.side_effect=[invalid,finish('skipped')]
        self.controller.run_once()
        self.assertEqual(self.notifications,[])
        self.assertEqual(ActionStore(self.path).get('a1').status,'COMPLETED')

    def test_waiting_can_record_blockage_without_repetitive_message(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('finish_action',{'outcome':'WAITING','reason':'Waiting for a previously requested user choice','task_status':'ACTIVE'})
        self.assertEqual(self.queue.pending_notifications(),[])
        self.assertEqual(TaskStore(self.path).get('t1').status,'ACTIVE')

    def test_task_cannot_be_completed_merely_by_scheduling(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('add_actions',{'actions':[{'exec_at':{'delay_seconds':60},'description':'Finish goal later'}]})
        tools.execute('send_message',{'message':'I’ve scheduled the next step.'})
        with self.assertRaises(ValueError):
            tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Only scheduled','task_status':'COMPLETED'})
        self.assertEqual(self.queue.pending_notifications(),[])
        self.assertEqual(ActionStore(self.path).get('a1').status,'RUNNING')
        self.assertEqual(TaskStore(self.path).get('t1').status,'ACTIVE')
