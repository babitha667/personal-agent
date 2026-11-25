"""Contextualizer creates one durable handoff and only initial task records."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock

from contextualizer import Contextualizer
from action_queue import ActionQueue
from database import connect
from message_queue import MessageQueue, MessageWorker
from models import Message
from planner_operations import PLANNER_TOOLS, apply_operations, initialize_planner
from storage import MessageStore
from task_store import TaskStore


def batch(*operations,content=''):
    return {'role':'assistant','content':content,'tool_calls':[
        {'id':f'c{i}','type':'function','function':{'name':name,'arguments':json.dumps(args)}}
        for i,(name,args) in enumerate(operations)]}


def create(description='Drink water',action_description='Arrange the requested water reminder',context=None,period=None):
    return batch(('create_action',{'task_id':None,'description':action_description,'context':context or {},
        'message_type':'new_task','new_task':{'description':description,'period':period}}))


def handoff(task_id=None,description='Respond naturally to the user',context=None,message_type='general_conversation'):
    return batch(('create_action',{'task_id':task_id,'description':description,'context':context or {},'message_type':message_type}))


class GenericPlannerTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db';self.model=Mock()
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model)
        self.queue=ActionQueue(self.path)

    def send(self,text,response):
        self.model.chat.return_value=response
        return self.contextualizer.receive_message(Message.from_content(text))

    def test_only_handoff_tool_is_exposed(self):
        result=self.send('Hey',handoff())
        self.assertEqual({t['function']['name'] for t in self.model.chat.call_args.kwargs['tools']},{'create_action'})
        self.assertEqual(result.tasks,[]);self.assertEqual(len(result.actions),1)
        self.assertIsNone(result.reply);self.assertEqual(self.queue.pending_notifications(),[])
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])

    def test_initial_task_and_exactly_one_kickoff_commit_together(self):
        start=datetime.now(timezone.utc)
        result=self.send('Remind me in ten minutes',create())
        self.assertEqual(len(result.tasks),1);self.assertEqual(len(result.actions),1)
        action=result.actions[0]
        self.assertEqual(action.task_id,result.tasks[0].task_id)
        self.assertEqual(action.exec_at,action.created_at)
        self.assertGreaterEqual(datetime.fromisoformat(action.exec_at),start)
        self.assertEqual(action.context['request']['message'],'Remind me in ten minutes')
        self.assertEqual(self.queue.pending_notifications(),[])

    def test_existing_task_update_only_enqueues_handoff(self):
        original=self.send('Remind me about meeting',create('Meeting'))
        with connect(self.path) as db:
            tasks=[tuple(r) for r in db.execute('SELECT * FROM tasks')]
            old=dict(db.execute('SELECT * FROM actions').fetchone())
        result=self.send('Take my notes to this meeting',handoff(original.tasks[0].task_id,
            'Evaluate the notes requirement and update the meeting action',message_type='additional_information'))
        with connect(self.path) as db:
            self.assertEqual(tasks,[tuple(r) for r in db.execute('SELECT * FROM tasks')])
            self.assertEqual(old,dict(db.execute('SELECT * FROM actions WHERE action_id=?',(old['action_id'],)).fetchone()))
        self.assertEqual(result.tasks,[]);self.assertEqual(len(result.actions),1)
        self.assertEqual(result.actions[0].task_id,original.tasks[0].task_id)

    def test_old_mutation_tools_and_multiple_handoffs_are_rejected(self):
        message=Message.from_content('Work');self.contextualizer.message_store.save(message)
        for operations in (
            [{'name':'create_task','arguments':{'description':'work'}}],
            [{'name':'update_task','arguments':{'task_id':'x','description':'changed'}}],
            [{'name':'cancel_task','arguments':{'task_id':'x'}}],
            [{'name':'create_action','arguments':handoff()['tool_calls'][0]['function']['arguments']}]*2,
        ):
            with self.assertRaises(ValueError):apply_operations(self.path,'u1',message,operations,[])
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])
        self.assertIsNone(self.queue.claim_due())

    def test_invalid_handoff_rolls_back_initial_task(self):
        response=create();args=json.loads(response['tool_calls'][0]['function']['arguments'])
        args['exec_at']='2099-01-01T00:00:00+00:00'
        message=Message.from_content('Later');self.contextualizer.message_store.save(message)
        with self.assertRaises(ValueError):apply_operations(self.path,'u1',message,[{'name':'create_action','arguments':args}],[])
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])

    def test_plain_reply_is_not_published_by_computer(self):
        self.model.chat.side_effect=[{'role':'assistant','content':'I will set it now.'},handoff()]
        result=self.contextualizer.receive_message(Message.from_content('Hi'))
        self.assertIsNone(result.reply)
        self.assertEqual(len(result.actions),1)
        self.assertEqual(self.queue.pending_notifications(),[])

    def test_duplicate_incoming_message_has_one_task_and_one_action(self):
        message=Message.from_content('Remind me');self.model.chat.return_value=create()
        first=self.contextualizer.receive_message(message);second=self.contextualizer.receive_message(message)
        self.assertEqual(first.actions,second.actions)
        self.assertEqual(self.model.chat.call_count,1)
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM actions').fetchone()[0],1)

    def test_inbox_done_only_when_task_and_handoff_persisted(self):
        inbox=MessageQueue(self.path,'u1');message=Message.from_content('Remind me')
        inbox.enqueue(message);self.model.chat.return_value=create()
        MessageWorker(inbox,self.contextualizer).run_once()
        self.assertEqual(inbox.status(message.id),'DONE')
        self.assertEqual(len(TaskStore(self.path).user_context('u1')),1)
        self.assertEqual([n['message'] for n in self.queue.pending_notifications()],['👍 ACK'])

    def test_task_selection_is_user_scoped(self):
        original=self.send('Work',create())
        message=Message.from_content('Change it');self.contextualizer.message_store.save(message,user_id='other')
        args=json.loads(handoff(original.tasks[0].task_id)['tool_calls'][0]['function']['arguments'])
        with self.assertRaises(ValueError):apply_operations(self.path,'other',message,[{'name':'create_action','arguments':args}],[])

    def test_initial_task_cannot_be_created_for_general_conversation(self):
        message=Message.from_content('Hey');self.contextualizer.message_store.save(message)
        args=json.loads(create()['tool_calls'][0]['function']['arguments']);args['message_type']='general_conversation'
        with self.assertRaises(ValueError):apply_operations(self.path,'u1',message,[{'name':'create_action','arguments':args}],[])

    def test_taskless_message_does_not_need_persistent_memory(self):
        self.send('What facewash did I mention?',handoff(context={'relevant_history':['I use COSRX']}))
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])
        with connect(self.path) as db:
            self.assertIsNone(db.execute("SELECT 1 FROM sqlite_master WHERE name='memories'").fetchone())
