import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue
from controller import ActionController
from database import connect, initialize_task_tables
from message_queue import MessageQueue, MessageWorker
from models import Message
from outbox import initialize_outbox,enqueue_outgoing
from storage import MessageStore

class OutboxTests(unittest.TestCase):
    def test_ack_and_plain_reply_are_durable_and_only_chat_drains_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'agent.db'
            model=Mock()
            from test_generic_planner import handoff,batch
            model.chat.return_value=handoff()
            contextualizer=Contextualizer(MessageStore(path),llm=model)
            inbox=MessageQueue(path,'u1')
            inbox.enqueue(Message.from_content('Hey'))
            direct=[]
            worker=MessageWorker(inbox,contextualizer)
            self.assertTrue(worker.run_once())
            self.assertEqual(direct,[])
            queue=ActionQueue(path)
            self.assertEqual([r['message'] for r in queue.pending_notifications()],['👍 ACK'])
            executor_model=Mock(chat=Mock(return_value=batch(('send_message',{'message':'Hello!'}),('finish_action',{'outcome':'COMPLETED','reason':'Answered'}))))
            standalone=ActionController(queue,ActionExecutor(queue,llm=executor_model),direct.append,deliver_to_terminal=False)
            standalone.run_once()
            self.assertEqual(len(queue.pending_notifications()),2)
            chat=[]
            listener=ActionController(queue,ActionExecutor(queue,llm=Mock()),chat.append,execute_actions=False)
            listener.run_once();listener.run_once()
            self.assertEqual(chat,['👍 ACK','Hello!'])
            with connect(path) as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM messages WHERE role='assistant'").fetchone()[0],1)

    def test_legacy_outbox_migration_preserves_rows_and_allows_chat(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'agent.db'
            initialize_task_tables(path)
            with connect(path) as db:
                db.execute('CREATE TABLE notifications(notification_id TEXT PRIMARY KEY,action_id TEXT NOT NULL REFERENCES actions(action_id),user_id TEXT NOT NULL,message TEXT NOT NULL,created_at TEXT NOT NULL,delivered_at TEXT)')
            from planning import validate_plan,save_plan
            tasks,actions=validate_plan([{'description':'Old','period':None,'actions':[{'exec_at':None,'description':'Notify'}]}],'u1')
            save_plan(path,tasks,actions)
            with connect(path) as db:
                db.execute('INSERT INTO notifications VALUES(?,?,?,?,?,NULL)',('old',actions[0].action_id,'u1','Old pending notification',actions[0].created_at))
            initialize_outbox(path)
            with connect(path) as db:
                self.assertEqual(db.execute("SELECT message FROM notifications WHERE notification_id='old'").fetchone()[0],'Old pending notification')
                enqueue_outgoing(db,'chat-1','u1','Hello')
                self.assertIsNone(db.execute("SELECT action_id FROM notifications WHERE notification_id='chat-1'").fetchone()[0])
                self.assertEqual(db.execute('PRAGMA foreign_key_check').fetchall(),[])
