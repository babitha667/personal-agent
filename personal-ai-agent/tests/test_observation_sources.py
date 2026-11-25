"""Local monitoring integrations, read-only tools, permissions and durable dedup."""
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from uuid import uuid4

from action_queue import ActionQueue
from action_store import ActionStore
from continuous_user_processor import ContinuousUserProcessor, from_environment
from database import connect
from email_store import EmailStore
from executor_tools import ExecutorTools
from models import Action, Message, Task
from observation_sources import EmailInboxSource, ReminderSource, local_sources
from proactive import ProactiveAgent
from storage import MessageStore
from task_store import TaskStore
from test_generic_planner import batch


class ObservationSourceTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db';self.queue=ActionQueue(self.path)
        self.messages=MessageStore(self.path);self.mail=EmailStore(self.path,'u1')
        self.now=datetime.now(timezone.utc)
        self.sources=local_sources(self.path)
        self.sources['reminders']=ReminderSource(self.path,clock=lambda:self.now)
        self.model=Mock()
        self.agent=ProactiveAgent(self.path,self.sources,llm=self.model)
        self.processor=ContinuousUserProcessor(self.path,self.agent)
        self.processor.configure_user('u1',sources=['email_inbox','reminders'],interval_seconds=1)

    def task(self,description='Attend a meeting',*,user='u1',status='ACTIVE',period=None,due=None,context=None):
        task=Task(str(uuid4()),user,description,status,period,self.now.isoformat())
        TaskStore(self.path).create(task)
        action=Action(str(uuid4()),task.task_id,(due or self.now+timedelta(minutes=10)).isoformat(),
                      'Carry out the user request','PENDING',0,self.now.isoformat(),context=context or {'bring':['notes']})
        ActionStore(self.path).create(action)
        return task,action

    def email_observation(self):
        return self.sources['email_inbox'].snapshot('u1')[0]

    def test_inbox_observes_read_and_unread_mail_without_changing_flags(self):
        incoming=self.mail.receive('colleague@example.test','The document is available.')
        first=self.email_observation()
        self.assertFalse(first.data['email']['read'])
        self.assertEqual(first.object_id,incoming['email_id'])
        self.assertEqual(first.key,self.email_observation().key)
        self.assertFalse(self.mail.read()[0]['read'])
        self.mail.read(mark_read=True)
        updated=self.email_observation()
        self.assertTrue(updated.data['email']['read'])
        self.assertNotEqual(first.key,updated.key)

    def test_snapshot_ignores_drafts_and_sends_but_thread_read_returns_them(self):
        task,action=self.task('Request access',due=self.now)
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        sent=tools.execute('send_email',{'to':'colleague@example.test','subject':'Access','content':'Please share access.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Sent; awaiting reply','task_status':'ACTIVE'})
        self.assertEqual(self.sources['email_inbox'].snapshot('u1'),[])
        reply=self.mail.receive('colleague@example.test','Granted!',thread_id=sent['thread_id'])
        observation=self.email_observation()
        self.assertEqual(observation.task_id,task.task_id)
        rows=self.sources['email_inbox'].read('u1',thread_id=sent['thread_id'])
        self.assertEqual([r['status'] for r in rows],['SENT','RECEIVED'])
        self.assertEqual([r['read'] for r in rows],[True,False])
        self.assertEqual(self.sources['email_inbox'].read('other',thread_id=sent['thread_id']),[])
        self.assertEqual(observation.data['email']['email_id'],reply['email_id'])

    def test_context_thread_reference_is_a_hint_not_keyword_matching(self):
        incoming=self.mail.receive('colleague@example.test','Here it is.')
        task,_=self.task('Get the design document',context={'email':{'thread_id':incoming['thread_id']}})
        self.assertEqual(self.email_observation().task_id,task.task_id)
        other,_=self.task('Another active request',context={'thread_id':incoming['thread_id']})
        observation=self.email_observation()
        self.assertIsNone(observation.task_id)
        self.assertEqual(set(observation.data['candidate_task_ids']),{task.task_id,other.task_id})

    def test_completed_task_is_not_automatically_linked_to_incoming_email(self):
        incoming=self.mail.receive('colleague@example.test','New information')
        self.task('Already handled',status='COMPLETED',context={'thread_id':incoming['thread_id']})
        self.assertIsNone(self.email_observation().task_id)

    def test_sources_and_connections_are_read_only(self):
        self.mail.receive('colleague@example.test','New information')
        task,action=self.task()
        with connect(self.path) as db:
            before=[tuple(r) for r in db.execute('SELECT * FROM actions')]
        self.sources['email_inbox'].snapshot('u1')
        self.sources['reminders'].snapshot('u1')
        with connect(self.path) as db:
            self.assertEqual(before,[tuple(r) for r in db.execute('SELECT * FROM actions')])
        self.assertFalse(self.mail.read()[0]['read'])
        with connect(self.path,read_only=True) as db:
            with self.assertRaises(sqlite3.OperationalError):db.execute("UPDATE emails SET read=1")

    def test_schedule_fingerprint_changes_only_with_state_and_schedule(self):
        task,action=self.task()
        first=self.sources['reminders'].snapshot('u1')[0]
        self.now+=timedelta(seconds=5)
        self.assertEqual(first.key,self.sources['reminders'].snapshot('u1')[0].key)
        with connect(self.path) as db:
            db.execute('UPDATE actions SET exec_at=? WHERE action_id=?',((self.now+timedelta(minutes=20)).isoformat(),action.action_id))
        self.assertNotEqual(first.key,self.sources['reminders'].snapshot('u1')[0].key)
        self.now+=timedelta(minutes=22)
        overdue=self.sources['reminders'].snapshot('u1')[0]
        self.assertEqual(overdue.data['pending_actions'][0]['timing_state'],'overdue')
        self.assertTrue(overdue.data['dispatcher_owns_delivery'])

    def test_administrative_and_proactive_handoffs_are_not_scheduled_reminders(self):
        self.task(context={'origin':'contextualizer','message_type':'new_task'})
        self.task(context={'origin':'proactive'})
        self.assertEqual(self.sources['reminders'].snapshot('u1'),[])

    def test_failed_task_is_observable_but_cannot_generate_active_task_action(self):
        task,_=self.task(status='FAILED')
        observation=self.sources['reminders'].snapshot('u1')[0]
        self.assertEqual(observation.data['task']['status'],'FAILED')
        self.model.chat.side_effect=[batch(('add_actions',{'actions':[{'observation_id':observation.key,'task_id':task.task_id,'description':'Retry'}]})),batch(('finish_check',{}))]
        self.assertEqual(self.processor.run_once(now=self.now),0)
        with connect(self.path) as db:self.assertEqual(db.execute('SELECT COUNT(*) FROM proactive_effects').fetchone()[0],0)

    def test_new_email_can_suggest_without_creating_task_and_unchanged_skips_llm(self):
        self.mail.receive('colleague@example.test','Would you like access to the design document?')
        observation=self.email_observation()
        self.model.chat.return_value=batch(('send_message',{'observation_id':observation.key,'message':'A colleague offered document access. Would you like me to follow up?'}),('finish_check',{}))
        self.assertEqual(self.processor.run_once(now=self.now),1)
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])
        self.assertEqual(len(self.queue.pending_notifications()),1)
        self.assertEqual(self.processor.run_once(now=self.now+timedelta(seconds=2)),0)
        self.assertEqual(self.model.chat.call_count,1)
        self.assertFalse(self.mail.read()[0]['read'])
        # A restarted observer reuses the persisted observation receipt.
        restarted=ContinuousUserProcessor(self.path,ProactiveAgent(self.path,local_sources(self.path),llm=self.model))
        self.assertEqual(restarted.run_once(now=self.now+timedelta(seconds=4)),0)
        self.assertEqual(self.model.chat.call_count,1)

    def test_semantic_relationship_can_trigger_an_unlinked_existing_task(self):
        task,action=self.task('Obtain the design document')
        self.mail.receive('colleague@example.test','Here is the design document access you asked for.')
        observation=self.email_observation();self.assertIsNone(observation.task_id)
        self.model.chat.return_value=batch(('add_actions',{'actions':[{'observation_id':observation.key,
            'task_id':task.task_id,'description':'Evaluate the access confirmation against the user request',
            'context':{'relationship':'Colleague supplied the requested design access'}}]}),('finish_check',{}))
        self.assertEqual(self.processor.run_once(now=self.now),1)
        with connect(self.path) as db:
            created=db.execute('SELECT * FROM actions WHERE action_id!=?',(action.action_id,)).fetchone()
        facts=json.loads(created['context'])
        self.assertEqual(created['task_id'],task.task_id)
        self.assertEqual(facts['object_id'],observation.object_id)
        self.assertEqual(facts['origin'],'proactive')
        self.assertIn('relationship',facts['relevant_context'])
        self.assertEqual(TaskStore(self.path).get(task.task_id).description,'Obtain the design document')
        self.assertEqual(self.processor.run_once(now=self.now+timedelta(seconds=2)),0)
        self.assertEqual(self.model.chat.call_count,1)

    def test_read_tools_include_complete_chat_actions_and_read_flags(self):
        task,action=self.task()
        for i in range(30):self.messages.save(Message.from_content(f'Personal context {i}'),'u1')
        self.messages.save(Message.from_content('Other user private chat'),'other')
        incoming=self.mail.receive('colleague@example.test','Please confirm the meeting.')
        def decide(messages,tools,timeout):
            initial=json.loads(messages[1]['content'])
            self.assertEqual(len(initial['chat_history']),30)
            self.assertEqual(initial['actions'][0]['context']['bring'],['notes'])
            self.assertNotIn('Other user private chat',json.dumps(initial))
            return batch(('read_email',{'thread_id':incoming['thread_id']}),('read_actions',{'task_id':task.task_id}),
                ('read_user_context',{}),('finish_check',{}))
        self.model.chat.side_effect=decide
        self.assertEqual(self.processor.run_once(now=self.now),0)
        self.assertFalse(self.mail.read()[0]['read'])

    def test_permissions_disable_mail_tool_and_read_other_user_is_empty(self):
        self.mail.receive('colleague@example.test','Private message')
        task,action=self.task()
        self.processor.configure_user('u1',sources=['reminders'],interval_seconds=1)
        observation=self.sources['reminders'].snapshot('u1')[0]
        def decide(messages,tools,timeout):
            self.assertNotIn('read_email',{t['function']['name'] for t in tools})
            self.assertNotIn('Private message',json.dumps(messages))
            return batch(('finish_check',{}))
        self.model.chat.side_effect=decide
        self.processor.run_once(now=self.now)
        self.assertEqual(self.sources['email_inbox'].snapshot('other'),[])

    def test_invalid_effect_batch_does_not_publish_partial_success(self):
        self.mail.receive('colleague@example.test','Please review')
        observation=self.email_observation()
        self.model.chat.side_effect=[batch(('send_message',{'observation_id':observation.key,'message':'Should not commit'}),
            ('add_actions',{'actions':[{'observation_id':observation.key,'task_id':'other-task','description':'Wrong owner'}]}),
            ('finish_check',{})),batch(('finish_check',{}))]
        self.assertEqual(self.processor.run_once(now=self.now),0)
        self.assertEqual(self.queue.pending_notifications(),[])

    def test_no_changes_make_no_model_call(self):
        self.processor.run_once(now=self.now)
        self.model.chat.assert_not_called()

    def test_configuration_preserves_disabled_sources_on_restart(self):
        self.processor.configure_user('u1',proactive_enabled=False,sources=[],interval_seconds=42)
        with patch.dict('os.environ',{},clear=True):restarted=from_environment(self.path,'u1')
        with connect(self.path) as db:config=dict(db.execute("SELECT * FROM continuous_users WHERE user_id='u1'").fetchone())
        self.assertEqual(config['proactive_enabled'],0)
        self.assertEqual(config['sources'],'[]')
        self.assertEqual(config['interval_seconds'],42)
        with patch.dict('os.environ',{'AGENT_PROACTIVE_SOURCES':'email_inbox,reminders','AGENT_PROACTIVE_ENABLED':'1'},clear=True):
            from_environment(self.path,'u1')
        with connect(self.path) as db:config=dict(db.execute("SELECT * FROM continuous_users WHERE user_id='u1'").fetchone())
        self.assertEqual(set(json.loads(config['sources'])),{'email_inbox','reminders'})
        self.assertEqual(config['proactive_enabled'],1)

    def test_unavailable_source_cannot_be_granted(self):
        with patch.dict('os.environ',{'AGENT_PROACTIVE_SOURCES':'gmail'},clear=True):
            with self.assertRaises(ValueError):from_environment(self.path,'u1')

    def test_cycle_can_handle_two_independent_emails(self):
        task,_=self.task('Obtain access')
        self.mail.receive('first@example.test','Would you like help organizing tomorrow?')
        self.mail.receive('second@example.test','Access to the requested design is now granted.')
        first,second=self.sources['email_inbox'].snapshot('u1')
        self.model.chat.return_value=batch(
            ('send_message',{'observation_id':first.key,'message':'Would you like help organizing tomorrow?'}),
            ('add_actions',{'actions':[{'observation_id':second.key,'task_id':task.task_id,
                'description':'Evaluate the access confirmation and continue the request'}]}),('finish_check',{}))
        self.assertEqual(self.processor.run_once(now=self.now),2)
        self.assertEqual(len(self.queue.pending_notifications()),1)
        self.assertEqual(len(TaskStore(self.path).user_context('u1')),1)
        self.assertEqual(self.processor.run_once(now=self.now+timedelta(seconds=2)),0)
        self.assertEqual(self.model.chat.call_count,1)

    def test_identical_configuration_does_not_invalidate_inflight_check(self):
        token,_=self.processor._claim('u1',self.now)
        self.processor.configure_user('u1',sources=['reminders','email_inbox'],interval_seconds=1)
        with connect(self.path) as db:row=db.execute("SELECT token FROM continuous_users WHERE user_id='u1'").fetchone()
        self.assertEqual(row['token'],token)
