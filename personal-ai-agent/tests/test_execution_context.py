import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue
from action_store import ActionStore
from database import connect, initialize_task_tables
from executor_tools import ExecutorTools, EXECUTOR_TOOLS
from models import Action, Message
from storage import MessageStore
from test_generic_planner import batch, create


class ExecutionContextTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db'
        self.model=Mock()
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model)
        self.queue=ActionQueue(self.path)

    def plan(self,delay=None):
        message=Message.from_content('Remind me to drink water'+(f' in {delay} seconds' if delay else ''))
        self.model.chat.return_value=create('Water','Process the requested water reminder',
            context={'activity':'drink water','preferences':{'temperature':'cold'}})
        return self.contextualizer.receive_message(message)

    def test_description_and_context_replace_instructions(self):
        result=self.plan()
        action=ActionStore(self.path).get(result.actions[0].action_id)
        self.assertEqual(action.description,'Process the requested water reminder')
        self.assertEqual(action.context['activity'],'drink water')
        self.assertNotIn('instructions',Action.__dataclass_fields__)
        with connect(self.path) as db:
            self.assertNotIn('instructions',[r['name'] for r in db.execute('PRAGMA table_info(actions)')])
            self.assertIsInstance(json.loads(db.execute('SELECT context FROM actions').fetchone()[0]),dict)

    def test_future_kickoff_schedules_and_delivery_uses_executor(self):
        for seconds in (120,300):
            with self.subTest(seconds=seconds):
                message=Message.from_content(f'Remind me to drink water in {seconds} seconds')
                self.model.chat.return_value=create(f'Water {seconds}','Handle the water reminder',
                    context={'activity':'drink water'})
                result=self.contextualizer.receive_message(message)
                initial=result.actions[0]
                due=datetime.fromisoformat(message.timestamp)+timedelta(seconds=seconds)
                self.assertLess(datetime.fromisoformat(initial.exec_at),due)
                self.assertEqual(initial.context['request']['message'],message.message)
                correct=batch(('add_actions',{'actions':[{'exec_at':{'delay_seconds':seconds},
                    'description':'Remind the user to drink water','context':{'activity':'drink water'}}]}),
                    ('send_message',{'message':f'I’ll remind you in {seconds//60} minutes.'}),
                    ('finish_action',{'outcome':'COMPLETED','reason':'Delivery scheduled','task_status':'ACTIVE'}))
                model=Mock();model.chat.return_value=correct
                executor=ActionExecutor(self.queue,llm=model)
                executor.execute(self.queue.claim_due())
                self.assertEqual(model.chat.call_count,1)
                self.assertIsNone(self.queue.claim_due(now=due-timedelta(seconds=1)))
                with connect(self.path) as db:
                    pending=db.execute("SELECT * FROM actions WHERE task_id=? AND status='PENDING'",(result.tasks[0].task_id,)).fetchone()
                    self.assertEqual(datetime.fromisoformat(pending['exec_at']),due)
                    self.assertEqual(db.execute('SELECT status FROM tasks WHERE task_id=?',(result.tasks[0].task_id,)).fetchone()[0],'ACTIVE')
                model.chat.return_value=batch(('send_message',{'message':f'Water reminder {seconds}'}),
                    ('finish_action',{'outcome':'COMPLETED','reason':'Delivered','task_status':'COMPLETED'}))
                with patch('action_queue.utc_now',return_value=due+timedelta(seconds=1)):
                    executor.execute(self.queue.claim_due())
                self.assertEqual(model.chat.call_count,2)
                payload=json.loads(model.chat.call_args.args[0][1]['content'])
                self.assertEqual(payload['claimed_action']['description'],'Remind the user to drink water')
                self.assertTrue(any(n['message']==f'Water reminder {seconds}' for n in self.queue.pending_notifications()))

    def test_followups_keep_entire_flat_history_with_committed_results(self):
        result=self.plan()
        expected_ids=[]
        for step in range(4):
            claim=self.queue.claim_due();expected_ids.append(claim.action.action_id)
            tools=ExecutorTools(self.queue,claim)
            tools.execute('add_actions',{'actions':[{'exec_at':{'immediate':True},'description':f'Next step {step+1}',
                'context':{'decisions':{f'step_{step}':'continue'}}}]})
            tools.execute('send_message',{'message':f'Progress {step}'})
            finished=tools.execute('finish_action',{'outcome':'COMPLETED','reason':f'Step {step} verified','task_status':'ACTIVE'})
            successor=ActionStore(self.path).get(finished['follow_up_action_ids'][0])
            history=successor.context['past_actions']
            self.assertEqual([a['action_id'] for a in history],expected_ids)
            self.assertEqual(len(history),step+1)
            self.assertEqual(history[-1]['outcome'],'COMPLETED')
            self.assertEqual(history[-1]['tool_results']['message'],f'Progress {step}')
            self.assertEqual(history[-1]['tool_results']['follow_up_action_ids'],[successor.action_id])
            self.assertEqual(successor.context['preferences'],{'temperature':'cold'})
            self.assertEqual(len(successor.context['decisions']),step+1)
            self.assertFalse(any('past_actions' in a['context'] for a in history))
            self.assertIsInstance(successor.context,dict)

    def test_legacy_instruction_column_migrates_without_losing_data(self):
        path=self.path.parent/'old.db'
        db=sqlite3.connect(path)
        db.executescript('''CREATE TABLE tasks(task_id TEXT PRIMARY KEY,user_id TEXT,description TEXT,status TEXT CHECK(status IN ('ACTIVE','COMPLETED','FAILED','CANCELLED')),period INTEGER,created_at TEXT,last_executed_at TEXT);
            CREATE TABLE actions(action_id TEXT PRIMARY KEY,task_id TEXT REFERENCES tasks(task_id),exec_at TEXT,instructions TEXT NOT NULL,status TEXT,attempt_count INTEGER,created_at TEXT);
            INSERT INTO tasks VALUES('t','u1','Existing task','ACTIVE',NULL,'2026-10-26T00:00:00+00:00',NULL);
            INSERT INTO actions VALUES('a','t','2026-10-26T00:00:00+00:00','Preserved prior step','PENDING',0,'2026-10-26T00:00:00+00:00');''')
        db.close();initialize_task_tables(path)
        action=ActionStore(path).get('a')
        self.assertEqual(action.description,'Preserved prior step')
        self.assertEqual(action.context['legacy_details'],'Preserved prior step')
        self.assertIsInstance(action.context,dict)
        with connect(path) as db:self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])

    def test_context_inputs_must_be_objects(self):
        self.plan()
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        with self.assertRaises(ValueError):
            tools.execute('add_actions',{'actions':[{'exec_at':{'immediate':True},'description':'Next','context':'{"facts":[]}' }]})
        self.assertEqual(tools.follow_ups,[])

    def test_claimed_kickoff_phase_is_explicit(self):
        self.plan(delay=120)
        claim=self.queue.claim_due()
        self.assertEqual(self.queue.context(claim)['phase'],'KICKOFF')
