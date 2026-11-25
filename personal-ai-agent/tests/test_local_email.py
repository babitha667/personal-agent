import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue, StaleClaim
from database import connect
from email_store import EmailStore, mailbox_address
from executor_tools import ExecutorTools
from models import Message
from storage import MessageStore
from test_generic_planner import batch, create


class LocalEmailTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db'
        self.queue=ActionQueue(self.path)
        self.model=Mock(chat=Mock(return_value=create('Request access','Ask colleague for document access',
            context={'recipient':'colleague@example.test','document':'Project notes'})))
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model)
        self.plan=self.contextualizer.receive_message(Message.from_content('Ask colleague@example.test for access to Project notes'))
        self.store=EmailStore(self.path,'u1')

    def stage(self,tools,thread=None):
        return tools.execute('send_email',{'to':'colleague@example.test','subject':'Document access request','content':'Could you grant me access to the Project notes document?', 'thread_id':thread})

    def send(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        staged=self.stage(tools)
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Local email saved','task_status':'COMPLETED'})
        return staged

    def test_email_and_identity_commit_with_action_history(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        staged=self.stage(tools)
        self.assertEqual(self.store.read(),[])
        self.assertTrue(staged['local_only'])
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Saved local email','task_status':'COMPLETED'})
        row=self.store.read()[0]
        self.assertEqual(row['user_id'],'u1');self.assertTrue(row['read'])
        self.assertEqual(row['from'],mailbox_address('u1'))
        self.assertEqual(row['to'],'colleague@example.test')
        self.assertEqual(row['email_id'],staged['email_id'])
        self.assertIn('timestamp',row)
        with connect(self.path) as db:
            import json
            result=json.loads(db.execute('SELECT details FROM action_results').fetchone()[0])
        self.assertEqual(result['send_email']['email_id'],row['email_id'])
        self.assertTrue(result['send_email']['local_only'])
        self.assertIsNone(self.queue.claim_due())

    def test_incoming_replies_append_and_start_unread(self):
        sent=self.send()
        first=self.store.receive('colleague@example.test','What access level do you need?',thread_id=sent['thread_id'])
        second=self.store.receive('colleague@example.test','I can grant view access.',thread_id=sent['thread_id'])
        rows=self.store.read(thread_id=sent['thread_id'])
        self.assertEqual(len(rows),3)
        self.assertEqual(len({r['email_id'] for r in rows}),3)
        self.assertTrue(rows[0]['read']);self.assertFalse(rows[1]['read']);self.assertFalse(rows[2]['read'])
        self.assertEqual(rows[1]['email_id'],first['email_id'])
        self.assertEqual(rows[2]['email_id'],second['email_id'])
        self.assertEqual([r['timestamp'] for r in rows],sorted(r['timestamp'] for r in rows))
        self.assertEqual(len(self.store.read(unread_only=True)),2)

    def test_read_status_is_explicit_and_inspection_is_read_only(self):
        incoming=self.store.receive('colleague@example.test','Reply content')
        rows=self.store.read(email_id=incoming['email_id'])
        self.assertFalse(rows[0]['read'])
        self.assertEqual(len(self.store.read(unread_only=True)),1)
        rows=self.store.read(thread_id=incoming['thread_id'],mark_read=True)
        self.assertTrue(rows[0]['read'])
        self.assertEqual(self.store.read(unread_only=True),[])

    def test_user_cannot_read_or_reply_to_other_mailbox(self):
        sent=self.send();other=EmailStore(self.path,'u2')
        self.assertEqual(other.read(email_id=sent['email_id']),[])
        with self.assertRaises(ValueError):other.read(thread_id=sent['thread_id'])
        with self.assertRaises(ValueError):other.receive('colleague@example.test','Private',thread_id=sent['thread_id'])
        self.assertEqual(len(self.store.read()),1)

    def test_sender_and_user_cannot_be_chosen_by_model(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        for field in ['from','user_id']:
            with self.assertRaises(ValueError):
                tools.execute('send_email',{'to':'colleague@example.test','subject':'Document access request','content':'Request','thread_id':None,field:'spoofed'})
        with self.assertRaises(ValueError):tools.execute('send_email',{'to':'invented','content':'Request','thread_id':None})
        self.assertIsNone(tools.email)

    def test_failed_batch_never_sends_or_publishes_success(self):
        bad=batch(('send_email',{'to':'colleague@example.test','subject':'Document access request','content':'Request','thread_id':None}),
            ('send_message',{'message':'Incorrect success'}),
            ('finish_action',{'outcome':'WAITING','reason':'Invalid pending send','task_status':'ACTIVE'}))
        model=Mock();model.chat.side_effect=[bad,batch(('finish_action',{'outcome':'SKIPPED','reason':'No effect committed','task_status':'ACTIVE'}))]
        ActionExecutor(self.queue,model).execute(self.queue.claim_due())
        self.assertEqual(self.store.read(),[])
        self.assertEqual(self.queue.pending_notifications(),[])

    def test_cancellation_fences_staged_email_and_retry_cannot_duplicate(self):
        claim=self.queue.claim_due();tools=ExecutorTools(self.queue,claim);self.stage(tools)
        with connect(self.path) as db:db.execute("UPDATE actions SET status='CANCELLED' WHERE action_id=?",(claim.action.action_id,))
        with self.assertRaises(StaleClaim):tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Send'})
        self.assertEqual(self.store.read(),[])
        self.queue.retry(claim)
        self.assertIsNone(self.queue.claim_due())

    def test_model_selects_tools_for_simple_send_in_one_response(self):
        response=batch(('send_email',{'to':'colleague@example.test','subject':'Document access request','content':'Please share access.','thread_id':None}),
            ('send_message',{'message':'Saved your access request in the local demo mailbox.'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Local email request saved','task_status':'COMPLETED'}))
        model=Mock(chat=Mock(return_value=response))
        ActionExecutor(self.queue,model).execute(self.queue.claim_due())
        self.assertEqual(len(self.store.read()),1)
        self.assertEqual(model.chat.call_count,1)
        self.assertEqual(len(self.queue.pending_notifications()),1)

    def test_unread_filter_applies_before_limit(self):
        old=self.store.receive('colleague@example.test','Old unread')
        for _ in range(3):
            message=self.store.receive('colleague@example.test','New read')
            self.store.read(email_id=message['email_id'],mark_read=True)
        self.assertEqual(self.store.read(unread_only=True,limit=1)[0]['email_id'],old['email_id'])

    def test_executor_mark_read_waits_for_atomic_completion(self):
        incoming=self.store.receive('colleague@example.test','Please clarify which document')
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        result=tools.execute('read_email',{'email_id':incoming['email_id'],'mark_read':True})
        self.assertTrue(result['read_update_staged'])
        self.assertFalse(self.store.read(email_id=incoming['email_id'])[0]['read'])
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Read local email','task_status':'COMPLETED'})
        self.assertTrue(self.store.read(email_id=incoming['email_id'])[0]['read'])

    def test_second_attempt_cannot_commit_same_action_email_twice(self):
        claim=self.queue.claim_due();tools=ExecutorTools(self.queue,claim);self.stage(tools)
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Saved'})
        with self.assertRaises(ValueError):self.stage(tools)
        with self.assertRaises(StaleClaim):self.queue.finish(claim,'completed',[],None,email=tools.email)
        self.assertEqual(len(self.store.read()),1)

    def draft(self,**extra):
        claim=self.queue.claim_due();tools=ExecutorTools(self.queue,claim)
        staged=tools.execute('draft_email',{'to':'colleague@example.test','content':'Please share Project notes.',**extra})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Unsent draft saved','task_status':'COMPLETED'})
        return staged

    def new_action(self,description='Send the prepared draft'):
        from test_generic_planner import create
        self.model.chat.return_value=create(description,description)
        self.contextualizer.receive_message(Message.from_content(description))
        return ExecutorTools(self.queue,self.queue.claim_due())

    def test_draft_is_unsent_and_sending_preserves_email_and_thread_ids(self):
        draft=self.draft(subject='Access request')
        row=self.store.read(email_id=draft['email_id'])[0]
        self.assertEqual(row['status'],'DRAFT')
        self.assertEqual(self.store.read(status='SENT'),[])
        tools=self.new_action()
        staged=tools.execute('send_email',{'email_id':draft['email_id']})
        self.assertEqual(staged['status'],'SENT')
        self.assertEqual(self.store.read(email_id=draft['email_id'])[0]['status'],'DRAFT')
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Draft sent in local mailbox','task_status':'COMPLETED'})
        rows=self.store.read()
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['email_id'],draft['email_id'])
        self.assertEqual(rows[0]['thread_id'],draft['thread_id'])
        self.assertEqual(rows[0]['status'],'SENT')
        self.assertEqual(rows[0]['subject'],'Access request')
        self.assertEqual(self.store.read(status='DRAFT'),[])

    def test_draft_can_be_edited_without_sending_and_subject_is_persisted(self):
        draft=self.draft()
        tools=self.new_action('Choose an appropriate draft subject')
        tools.execute('draft_email',{'email_id':draft['email_id'],'subject':'Request for Project Notes Access'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Draft subject chosen','task_status':'COMPLETED'})
        row=self.store.read()[0]
        self.assertEqual(row['status'],'DRAFT')
        self.assertEqual(row['subject'],'Request for Project Notes Access')
        self.assertEqual(len(self.store.read()),1)
        with connect(self.path) as db:
            import json
            result=json.loads(db.execute('SELECT details FROM action_results ORDER BY completed_at DESC LIMIT 1').fetchone()[0])
        self.assertEqual(result['draft_email']['subject'],row['subject'])
        self.assertEqual(result['draft_email']['status'],'DRAFT')

    def test_sent_email_cannot_be_edited_or_sent_again_by_id(self):
        sent=self.send();tools=self.new_action()
        with self.assertRaisesRegex(ValueError,'already sent'):
            tools.execute('send_email',{'email_id':sent['email_id']})
        with self.assertRaisesRegex(ValueError,'already sent'):
            tools.execute('draft_email',{'email_id':sent['email_id'],'subject':'Changed'})
        self.assertEqual(len(self.store.read()),1)

    def test_failed_draft_send_keeps_draft_unsent(self):
        draft=self.draft(subject='Access request')
        tools=self.new_action()
        tools.execute('send_email',{'email_id':draft['email_id']})
        with self.assertRaises(ValueError):
            tools.execute('finish_action',{'outcome':'WAITING','reason':'Invalid staged send','task_status':'ACTIVE'})
        self.assertEqual(self.store.read()[0]['status'],'DRAFT')

    def test_concurrent_draft_edit_cannot_be_overwritten_by_stale_send(self):
        draft=self.draft(subject='Original')
        tools=self.new_action();tools.execute('send_email',{'email_id':draft['email_id']})
        with connect(self.path) as db:
            db.execute('UPDATE emails SET subject=?,revision=revision+1 WHERE email_id=?',('New subject',draft['email_id']))
        with self.assertRaisesRegex(ValueError,'Draft changed'):
            tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Stale send'})
        row=self.store.read()[0]
        self.assertEqual(row['status'],'DRAFT');self.assertEqual(row['subject'],'New subject')

    def test_verified_draft_result_and_recent_question_are_planner_input(self):
        draft=self.draft(subject='Access request')
        self.contextualizer.message_store.save(Message.from_content('Would you like a different subject?'),'u1',role='assistant')
        from test_generic_planner import handoff
        self.model.chat.return_value=handoff(description='Resolve the delegated draft subject using recent dialogue')
        self.contextualizer.receive_message(Message.from_content('You decide'))
        import json
        payload=json.loads(self.model.chat.call_args.args[0][0]['content'].split('Current context:\n',1)[1])
        self.assertEqual(payload['recent_dialogue'][-1]['message'],'Would you like a different subject?')
        result=payload['tasks'][0]['last_result']['details']['draft_email']
        self.assertEqual(result['status'],'DRAFT');self.assertEqual(result['email_id'],draft['email_id'])

    def test_incomplete_draft_cannot_be_sent_until_address_and_subject_exist(self):
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        draft=tools.execute('draft_email',{'content':'Please share access to the Project notes document.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Unsent draft saved','task_status':'COMPLETED'})
        sender=self.new_action()
        with self.assertRaises(ValueError):sender.execute('send_email',{'email_id':draft['email_id']})
        self.assertEqual(self.store.read()[0]['status'],'DRAFT')
        sender.execute('send_email',{'email_id':draft['email_id'],'to':'colleague@example.test','subject':'Project notes access'})
        sender.execute('finish_action',{'outcome':'COMPLETED','reason':'Draft addressed and sent','task_status':'COMPLETED'})
        self.assertEqual(self.store.read()[0]['status'],'SENT')
        self.assertEqual(len(self.store.read()),1)

    def test_old_mail_migrates_as_sent_or_received_never_draft(self):
        import sqlite3
        old=self.path.parent/'old-mail.db'
        db=sqlite3.connect(old)
        db.executescript('''CREATE TABLE emails(sequence INTEGER PRIMARY KEY AUTOINCREMENT,email_id TEXT UNIQUE NOT NULL,
            thread_id TEXT NOT NULL,user_id TEXT NOT NULL,timestamp TEXT NOT NULL,"from" TEXT NOT NULL,"to" TEXT NOT NULL,
            content TEXT NOT NULL,read INTEGER NOT NULL,action_id TEXT UNIQUE);
            INSERT INTO emails(email_id,thread_id,user_id,timestamp,"from","to",content,read,action_id)
                VALUES('sent','thread','u1','2026-10-26T00:00:00+00:00','me@example.test','colleague@example.test','Old send',1,'old-action');
            INSERT INTO emails(email_id,thread_id,user_id,timestamp,"from","to",content,read,action_id)
                VALUES('reply','thread','u1','2026-10-26T00:01:00+00:00','colleague@example.test','me@example.test','Old reply',0,NULL);''')
        db.close()
        rows=EmailStore(old,'u1').read()
        self.assertEqual([r['status'] for r in rows],['SENT','RECEIVED'])
        self.assertEqual([r['subject'] for r in rows],['',''])
        self.assertEqual([r['content'] for r in rows],['Old send','Old reply'])
