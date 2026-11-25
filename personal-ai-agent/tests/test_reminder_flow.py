import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from legacy_contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue
from controller import ActionController
from database import connect
from models import Message
from storage import MessageStore
from test_contextualizer import call, TEST_TOOLS


class ReminderFlowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'agent.db'
        self.store = MessageStore(self.path)
        self.model = Mock()
        self.contextualizer = Contextualizer(self.store,llm=self.model, tool_definitions=TEST_TOOLS)
        self.queue = ActionQueue(self.path)

    def plan(self, notification='Drink water!', period=None):
        self.model.chat.return_value=call('apply_plan',{'tasks':[{
            'description':'Drink water','period':period,'actions':[{
                'exec_at':{'delay_seconds':300},'description':'Remind user to drink water',
                'notification':notification}]}],'cancel_task_ids':[]})
        message=Message.from_content('Remind me in 5 minutes')
        return self.contextualizer.receive_message(message), datetime.fromisoformat(message.timestamp)+timedelta(minutes=5)

    def test_initial_now_executor_confirmation_then_due_delivery_in_chat(self):
        result,due=self.plan()
        initial=result.actions[0]
        self.assertEqual(initial.exec_at,initial.created_at)
        self.assertEqual(initial.context['requested_time'],due.isoformat())
        self.assertIsNone(result.reply)
        from mock_execution import executor_model
        model=executor_model()
        standalone_output=[]
        standalone=ActionController(self.queue,ActionExecutor(self.queue,llm=model),standalone_output.append,
                                    deliver_to_terminal=False)
        self.assertTrue(standalone.run_once())
        self.assertEqual(standalone_output,[])
        self.assertIsNone(self.queue.claim_due())
        with connect(self.path) as db:
            pending=dict(db.execute("SELECT * FROM actions WHERE status='PENDING'").fetchone())
            self.assertEqual(pending['exec_at'],due.isoformat())
            self.assertEqual(pending['description'],'Remind user to drink water')
            self.assertNotIn('notification',pending)
        chat=[]
        listener=ActionController(self.queue,ActionExecutor(self.queue,llm=model),chat.append,execute_actions=False)
        self.assertFalse(listener.run_once())
        self.assertEqual(chat,['Scheduled the reminder.'])
        with patch('action_queue.utc_now',return_value=due+timedelta(seconds=1)):
            self.assertTrue(standalone.run_once())
        listener.run_once()
        self.assertEqual(chat,['Scheduled the reminder.','Drink water!'])
        self.assertEqual(model.chat.call_count,2)
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT status FROM tasks').fetchone()[0],'COMPLETED')
            self.assertEqual(db.execute("SELECT COUNT(*) FROM messages WHERE role='assistant'").fetchone()[0],2)

    def test_executor_schedules_description_for_later_execution(self):
        result,due=self.plan(notification=None)
        from mock_execution import executor_model
        model=executor_model()
        model.chat.side_effect=None
        model.chat.return_value={'role':'assistant','content':'','tool_calls':[
            *call('add_actions',{'actions':[{'exec_at':due.isoformat(),'description':'Deliver water reminder'}]})['tool_calls'],
            *call('send_message',{'message':'Scheduled the reminder.'})['tool_calls'],
            *call('finish_action',{'outcome':'COMPLETED','reason':'Delivery scheduled'})['tool_calls']]}

        output=[]
        controller=ActionController(self.queue,ActionExecutor(self.queue,llm=model),output.append)
        controller.run_once()
        self.assertEqual(output,['Scheduled the reminder.'])
        self.assertEqual(model.chat.call_count,1)
        with connect(self.path) as db:
            row=db.execute("SELECT exec_at,description FROM actions WHERE status='PENDING'").fetchone()
            self.assertEqual(tuple(row),(due.isoformat(),'Deliver water reminder'))

    def test_recurrence_uses_delivery_time_not_initial_now(self):
        result,due=self.plan(period=86400)
        from mock_execution import executor_model
        executor=ActionExecutor(self.queue,llm=executor_model())
        executor.execute(self.queue.claim_due())
        with patch('action_queue.utc_now',return_value=due+timedelta(seconds=1)):
            executor.execute(self.queue.claim_due())
        with connect(self.path) as db:
            row=db.execute("SELECT exec_at,description FROM actions WHERE status='PENDING'").fetchone()
            self.assertIsNone(row)
            task=db.execute('SELECT status,last_executed_at FROM tasks').fetchone()
            self.assertEqual(task['status'],'ACTIVE')
            self.assertEqual(datetime.fromisoformat(task['last_executed_at']),due+timedelta(seconds=1))
