"""The planner preserves intent; executor alone schedules and delivers actions."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue, StaleClaim
from action_store import ActionStore
from database import connect, initialize_task_tables
from executor_tools import EXECUTOR_TOOLS, ExecutorTools
from models import Message
from planner_operations import PLANNER_TOOLS
from storage import MessageStore
from test_generic_planner import batch, create

class ActionTimingFlowTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db'
        self.model=Mock()
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model)
        self.queue=ActionQueue(self.path)

    def plan(self, text):
        self.model.chat.return_value=create('Water','Arrange the requested water reminder')
        return self.contextualizer.receive_message(Message.from_content(text))

    def test_tool_schemas_separate_planning_from_timing(self):
        for tool in PLANNER_TOOLS:
            self.assertNotIn('requested_execution',tool['function']['parameters']['properties'])
        add=next(t for t in EXECUTOR_TOOLS if t['function']['name']=='add_actions')
        self.assertEqual(set(add['function']['parameters']['properties']['actions']['items']['properties']),
                         {'description','context','exec_at'})
        result=self.plan('Remind me at 6 PM')
        claim=self.queue.claim_due()
        tools=ExecutorTools(self.queue,claim)
        with self.assertRaises(ValueError):
            tools.execute('add_actions',{'actions':[{'exec_at':{'immediate':True},'description':'Water','notification':'Water!'}]})
        self.assertEqual(result.actions[0].context['request']['message'],'Remind me at 6 PM')

    def test_reschedule_is_a_handoff_then_executor_replaces_delivery(self):
        result=self.plan('Remind me to drink water in five minutes')
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('add_actions',{'actions':[{'exec_at':{'delay_seconds':300},'description':'Remind the user to drink water'}]})
        tools.execute('send_message',{'message':'Scheduled.'})
        finished=tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Scheduled','task_status':'ACTIVE'})
        old=ActionStore(self.path).get(finished['follow_up_action_ids'][0])
        from test_generic_planner import handoff
        message=Message.from_content('Make that two minutes from now')
        self.model.chat.return_value=handoff(result.tasks[0].task_id,'Evaluate the new timing and reschedule',message_type='task_update')
        update=self.contextualizer.receive_message(message)
        self.assertEqual(ActionStore(self.path).get(old.action_id).status,'PENDING')
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        due=datetime.fromisoformat(message.timestamp)+timedelta(minutes=2)
        replacement=tools.execute('update_action',{'action_id':old.action_id,'exec_at':due.isoformat()})
        tools.execute('send_message',{'message':'Moved it to two minutes from now.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Rescheduled','task_status':'ACTIVE'})
        self.assertEqual(ActionStore(self.path).get(old.action_id).status,'CANCELLED')
        self.assertEqual(datetime.fromisoformat(ActionStore(self.path).get(replacement['replacement_action_id']).exec_at),due)
        self.assertEqual(update.tasks,[])

    def test_legacy_columns_migrate_facts_and_preserve_links(self):
        self.plan('Remind me later')
        claim=self.queue.claim_due()
        with connect(self.path) as db:
            db.execute('ALTER TABLE actions ADD COLUMN scheduled_for TEXT')
            db.execute('ALTER TABLE actions ADD COLUMN notification TEXT')
            db.execute('UPDATE actions SET scheduled_for=?,notification=?',('2099-01-01T18:00:00+05:30','Drink cold water'))
        initialize_task_tables(self.path)
        initialize_task_tables(self.path)
        with connect(self.path) as db:
            fields={r['name'] for r in db.execute('PRAGMA table_info(actions)')}
            self.assertNotIn('scheduled_for',fields);self.assertNotIn('notification',fields)
            self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
            self.assertEqual(db.execute('SELECT token FROM action_leases').fetchone()[0],claim.token)
        action=ActionStore(self.path).get(claim.action.action_id)
        self.assertEqual(action.context['requested_time'],'2099-01-01T18:00:00+05:30')
        self.assertEqual(action.context['legacy_message'],'Drink cold water')

    def test_absolute_followup_uses_aware_exec_at(self):
        self.plan('Remind me to drink water tomorrow at 6 PM IST')
        claim=self.queue.claim_due()
        due=(datetime.now(timezone(timedelta(hours=5,minutes=30)))+timedelta(days=1)).replace(hour=18,minute=0,second=0,microsecond=0)
        tools=ExecutorTools(self.queue,claim)
        tools.execute('add_actions',{'actions':[{'description':'Remind the user to drink water','exec_at':due.isoformat()}]})
        tools.execute('send_message',{'message':'I’ll remind you tomorrow at 6 PM IST.'})
        result=tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Scheduled','task_status':'ACTIVE'})
        follow=ActionStore(self.path).get(result['follow_up_action_ids'][0])
        self.assertEqual(datetime.fromisoformat(follow.exec_at),due)
        self.assertEqual(follow.context['past_actions'][-1]['outcome'],'COMPLETED')

    def test_enrichment_queued_before_kickoff_preserves_delivery_anchor(self):
        initial=self.plan('Remind me to order coffee in five minutes')
        from test_generic_planner import handoff
        self.model.chat.return_value=handoff(initial.tasks[0].task_id,'Evaluate the decaf preference',
            context={'preference':'decaf'},message_type='additional_information')
        self.contextualizer.receive_message(Message.from_content('I prefer decaf coffee'))
        first=ExecutorTools(self.queue,self.queue.claim_due())
        first.execute('add_actions',{'actions':[{'exec_at':{'delay_seconds':300},'description':'Remind the user to order coffee'}]})
        first.execute('send_message',{'message':'Scheduled.'})
        saved=first.execute('finish_action',{'outcome':'COMPLETED','reason':'Scheduled','task_status':'ACTIVE'})
        delivery=ActionStore(self.path).get(saved['follow_up_action_ids'][0])
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        replacement=tools.execute('update_action',{'action_id':delivery.action_id,'context':{'preference':'decaf'}})
        tools.execute('send_message',{'message':'Noted.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Updated','task_status':'ACTIVE'})
        due=datetime.fromisoformat(initial.incoming_message.timestamp)+timedelta(minutes=5)
        self.assertEqual(datetime.fromisoformat(ActionStore(self.path).get(replacement['replacement_action_id']).exec_at),due)
