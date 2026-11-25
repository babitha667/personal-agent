import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timedelta
from unittest.mock import Mock
from legacy_contextualizer import Contextualizer
from agent_tools import PLANNER_TOOLS
from action_executor import ActionExecutor
from action_queue import ActionQueue
from controller import ActionController
from database import connect
from models import Message
from storage import MessageStore
from test_contextualizer import call

class NativeReminderTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path=Path(directory.name)/'agent.db'
        self.model=Mock()
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model,tool_definitions=PLANNER_TOOLS)
        self.queue=ActionQueue(self.path)
        from mock_execution import executor_model
        self.controller=ActionController(self.queue,ActionExecutor(self.queue,llm=executor_model()),lambda text:None)

    def test_native_create_converts_units_in_python_then_updates_and_cancels(self):
        self.model.chat.return_value=call('create_reminder',{'message':'Time to eat.','delay':5,'unit':'minutes'})
        message=Message.from_content('Can you set a reminder to eat in 5 minutes?')
        result=self.contextualizer.receive_message(message)
        self.assertEqual(result.actions[0].context.get('requested_time'),(datetime.fromisoformat(message.timestamp)+timedelta(minutes=5)).isoformat())
        self.controller.run_once()
        old_id=result.tasks[0].task_id
        due=result.actions[0].context.get('requested_time')
        self.model.chat.return_value=call('update_reminder',{'task_id':old_id,'message':'Time to eat lunch.'})
        updated=self.contextualizer.receive_message(Message.from_content('Make that lunch'))
        self.assertEqual(updated.cancelled_task_ids,[old_id])
        self.assertEqual(updated.actions[0].context.get('requested_time'),due)
        self.assertEqual(updated.actions[0].description,'Time to eat lunch.')
        self.model.chat.return_value=call('cancel_task',{'task_id':updated.tasks[0].task_id})
        cancelled=self.contextualizer.receive_message(Message.from_content('Cancel it'))
        self.assertEqual(cancelled.cancelled_task_ids,[updated.tasks[0].task_id])
        with connect(self.path) as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM tasks WHERE status='ACTIVE'").fetchone()[0],0)

    def test_native_missing_time_clarifies_and_followup_reaches_model_as_conversation(self):
        self.model.chat.return_value={'role':'assistant','content':'When would you like the reminder?'}
        self.contextualizer.receive_message(Message.from_content('Remind me to stretch'))
        self.model.chat.return_value=call('create_reminder',{'message':'Stretch!','delay':30,'unit':'seconds'})
        result=self.contextualizer.receive_message(Message.from_content('In 30 seconds'))
        self.assertEqual(len(result.tasks),1)
        messages=self.model.chat.call_args.args[0]
        self.assertEqual(messages[1],{'role':'user','content':'Remind me to stretch'})
        self.assertEqual(messages[2],{'role':'assistant','content':'When would you like the reminder?'})

    def test_invalid_delay_cannot_create_work(self):
        self.model.chat.side_effect=[call('create_reminder',{'message':'Eat!','delay':0,'unit':'minutes'}),
            {'role':'assistant','content':'When should I remind you?'}]
        result=self.contextualizer.receive_message(Message.from_content('Remind me to eat'))
        self.assertEqual(result.tasks,[])
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],0)

    def test_text_tool_cannot_silently_change_schedule(self):
        self.model.chat.return_value=call('create_reminder',{'message':'Water!','delay':5,'unit':'minutes'})
        original=self.contextualizer.receive_message(Message.from_content('Remind me about water in 5 minutes'))
        self.controller.run_once()
        self.model.chat.side_effect=[call('update_reminder',{'message':'Water with electrolytes!','delay':4,'unit':'minutes'}),
                                    call('update_reminder',{'message':'Water with electrolytes!'})]
        result=self.contextualizer.receive_message(Message.from_content('Add electrolytes to that reminder'))
        self.assertEqual(result.actions[0].context.get('requested_time'),original.actions[0].context.get('requested_time'))
        self.assertIn('electrolytes',result.actions[0].context.get('legacy_message'))
        schema=next(t for t in self.model.chat.call_args.kwargs['tools'] if t['function']['name']=='update_reminder')
        self.assertEqual(set(schema['function']['parameters']['properties']),{'message'})

    def test_time_tool_preserves_message_and_uses_bound_single_task(self):
        self.model.chat.return_value=call('create_reminder',{'message':'Water!','delay':5,'unit':'minutes'})
        original=self.contextualizer.receive_message(Message.from_content('Water in five minutes'))
        self.controller.run_once()
        self.model.chat.return_value=call('reschedule_reminder',{'delay':10,'unit':'minutes'})
        message=Message.from_content('Make it ten minutes instead')
        result=self.contextualizer.receive_message(message)
        self.assertEqual(result.cancelled_task_ids,[original.tasks[0].task_id])
        self.assertEqual(result.actions[0].context.get('legacy_message'),'Water!')
        self.assertEqual(result.actions[0].context.get('requested_time'),(datetime.fromisoformat(message.timestamp)+timedelta(minutes=10)).isoformat())

    def test_activity_payload_has_no_model_date_arithmetic(self):
        self.model.chat.return_value=call('create_reminder',{'activity':'drink water','delay':5,'unit':'minutes'})
        message=Message.from_content('Remind me to drink water in five minutes')
        result=self.contextualizer.receive_message(message)
        self.assertEqual(result.actions[0].context.get('legacy_message'),'Reminder: drink water.')
        self.assertEqual(result.actions[0].context.get('requested_time'),(datetime.fromisoformat(message.timestamp)+timedelta(minutes=5)).isoformat())

    def test_multiple_task_handles_are_scoped_to_active_user_tasks(self):
        self.model.chat.return_value=call('create_reminder',{'activity':'stretch','delay':5,'unit':'minutes'})
        stretch=self.contextualizer.receive_message(Message.from_content('Stretch in five minutes'))
        self.controller.run_once()
        self.model.chat.return_value=call('create_reminder',{'activity':'drink water','delay':6,'unit':'minutes'})
        water=self.contextualizer.receive_message(Message.from_content('Water in six minutes'))
        self.controller.run_once()
        self.model.chat.return_value=call('cancel_task',{'task_id':'r1','confirm':True})
        cancelled=self.contextualizer.receive_message(Message.from_content('Cancel the stretch reminder'))
        self.assertEqual(cancelled.cancelled_task_ids,[stretch.tasks[0].task_id])
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT status FROM tasks WHERE task_id=?',(water.tasks[0].task_id,)).fetchone()[0],'ACTIVE')

    def test_pending_activity_survives_restart_and_bound_time_completes_it(self):
        self.model.chat.return_value=call('ask_reminder_time',{'activity':'drink water'})
        result=self.contextualizer.receive_message(Message.from_content('Remind me to drink water'))
        self.assertEqual(result.reply,'When should I remind you to drink water?')
        with connect(self.path) as db:
            row=db.execute("SELECT pending_activity FROM messages WHERE role='assistant'").fetchone()
            self.assertEqual(row[0],'drink water')
        new_model=Mock()
        new_model.chat.return_value=call('set_reminder_time',{'delay':2,'unit':'minutes'})
        restarted=Contextualizer(MessageStore(self.path),llm=new_model,tool_definitions=PLANNER_TOOLS)
        message=Message.from_content('In two minutes')
        completed=restarted.receive_message(message)
        self.assertEqual(completed.actions[0].context.get('legacy_message'),'Reminder: drink water.')
        self.assertEqual(completed.actions[0].context.get('requested_time'),(datetime.fromisoformat(message.timestamp)+timedelta(minutes=2)).isoformat())
        self.assertIn('set_reminder_time',[t['function']['name'] for t in new_model.chat.call_args.kwargs['tools']])

    def test_empty_native_response_is_retried_without_failing_greeting(self):
        from llm import EmptyModelResponse
        self.model.chat.side_effect=[EmptyModelResponse('No final answer'),{'role':'assistant','content':'Hello!'}]
        result=self.contextualizer.receive_message(Message.from_content('Hey'))
        self.assertEqual(result.reply,'Hello!')
        self.assertEqual(self.model.chat.call_count,2)
        self.assertEqual(result.tasks,[])

    def test_pending_time_accepts_matching_redundant_activity_but_cannot_change_it(self):
        self.model.chat.return_value=call('ask_reminder_time',{'activity':'drink water'})
        self.contextualizer.receive_message(Message.from_content('Remind me to drink water'))
        self.model.chat.return_value=call('set_reminder_time',{'activity':'drink water','delay':2,'unit':'minutes'})
        result=self.contextualizer.receive_message(Message.from_content('In two minutes'))
        self.assertEqual(result.actions[0].context.get('legacy_message'),'Reminder: drink water.')

    def test_production_rejects_tools_not_exposed_to_model(self):
        from test_contextualizer import plan
        self.model.chat.side_effect=[call('apply_plan',plan()),{'role':'assistant','content':'I cannot use that tool.'}]
        result=self.contextualizer.receive_message(Message.from_content('Hey'))
        self.assertEqual(result.tasks,[])
        self.assertIn('Unknown tool: apply_plan',str(self.model.chat.call_args.args[0]))

    def test_notification_does_not_replace_the_pending_clarification(self):
        from planning import validate_plan,save_plan
        self.model.chat.return_value=call('ask_reminder_time',{'activity':'drink water'})
        self.contextualizer.receive_message(Message.from_content('Remind me to drink water'))
        tasks,actions=validate_plan([{'description':'Stretch','period':None,'actions':[{
            'exec_at':None,'description':'Stretch!','notification':'Stretch!'}]}],'u1')
        save_plan(self.path,tasks,actions)
        self.controller.run_once()
        with connect(self.path) as db:
            row=db.execute("SELECT kind FROM messages WHERE message='Stretch!' AND role='assistant'").fetchone()
            self.assertEqual(row[0],'notification')
        self.model.chat.return_value=call('set_reminder_time',{'delay':2,'unit':'minutes'})
        result=self.contextualizer.receive_message(Message.from_content('In two minutes'))
        self.assertEqual(result.actions[0].context.get('legacy_message'),'Reminder: drink water.')
        self.assertIn('set_reminder_time',[t['function']['name'] for t in self.model.chat.call_args.kwargs['tools']])
