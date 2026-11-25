import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock
from datetime import datetime,timedelta,timezone

from legacy_contextualizer import Contextualizer
from agent_tools import PLANNER_TOOLS
from action_queue import ActionQueue
from action_executor import ActionExecutor
from controller import ActionController
from database import connect
from models import Message
from test_contextualizer import call
from storage import MessageStore


class TaskContextTests(unittest.TestCase):
    def setUp(self):
        directory=tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path=Path(directory.name)/'agent.db'
        self.model=Mock()
        self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model,tool_definitions=PLANNER_TOOLS)
        self.queue=ActionQueue(self.path)
        from mock_execution import executor_model
        self.executor=ActionExecutor(self.queue,llm=executor_model())
        self.controller=ActionController(self.queue,self.executor,lambda text:None)
        self.model.chat.return_value=call('create_reminder',{'activity':'eat prawns','delay':5,'unit':'minutes'})
        self.original=self.contextualizer.receive_message(Message.from_content('Remind me to eat prawns in five minutes'))

    def enrich(self):
        self.model.chat.return_value=call('add_task_context',{'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'Thanks for letting me know.'})
        return self.contextualizer.receive_message(Message.from_content('I have prawns allergy'))

    def test_additional_info_enriches_pending_action_without_rescheduling(self):
        self.controller.run_once()
        with connect(self.path) as db:
            before=dict(db.execute("SELECT * FROM actions WHERE status='PENDING'").fetchone())
        result=self.enrich()
        self.assertEqual(result.tasks,[])
        with connect(self.path) as db:
            after=dict(db.execute('SELECT * FROM actions WHERE action_id=?',(before['action_id'],)).fetchone())
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM actions').fetchone()[0],2)
        self.assertEqual(after['exec_at'],before['exec_at'])
        self.assertIn('I have prawns allergy',after['description'])
        self.assertIn('I have prawns allergy',__import__('json').loads(after['context'])['user_information'][0])
        self.assertEqual(self.queue.pending_notifications()[0]['message'],'Thanks for letting me know.')

    def test_context_survives_race_with_already_claimed_initial_action(self):
        claim=self.queue.claim_due()
        self.enrich()
        self.executor.execute(claim)
        with connect(self.path) as db:
            followup=db.execute("SELECT * FROM actions WHERE status='PENDING'").fetchone()
            self.assertIn('I have prawns allergy',followup['description'])
            self.assertIn('I have prawns allergy',__import__('json').loads(followup['context'])['user_information'][0])
            self.assertEqual(followup['exec_at'],self.original.actions[0].context['requested_time'])

    def test_context_survives_race_with_delivery_claim(self):
        self.controller.run_once()
        due=datetime.fromisoformat(self.original.actions[0].context['requested_time'])+timedelta(seconds=1)
        claim=self.queue.claim_due(now=due)
        self.enrich()
        self.executor.execute(claim)
        notices=self.queue.pending_notifications()
        self.assertTrue(any('prawns allergy' in row['message'] for row in notices))

    def test_duplicate_creation_is_blocked_and_can_recover_as_context_update(self):
        self.controller.run_once()
        self.model.chat.side_effect=[call('create_reminder',{'activity':'eat prawns','delay':5,'unit':'minutes'}),
                                    call('add_task_context',{'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'Thanks for letting me know.'})]
        result=self.contextualizer.receive_message(Message.from_content('I have prawns allergy'))
        self.assertEqual(result.tasks,[])
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
            self.assertIn('prawns allergy',db.execute("SELECT description FROM actions WHERE status='PENDING'").fetchone()[0])

    def test_same_message_context_is_idempotent(self):
        self.controller.run_once()
        message=Message.from_content('I have prawns allergy')
        self.model.chat.return_value=call('add_task_context',{'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'Thanks for letting me know.'})
        self.contextualizer.receive_message(message)
        self.contextualizer.receive_message(message)
        with connect(self.path) as db:
            description=db.execute("SELECT description FROM actions WHERE status='PENDING'").fetchone()[0]
            self.assertEqual(description.count('I have prawns allergy'),1)

    def test_explicit_additional_identical_reminder_is_supported(self):
        self.model.chat.return_value=call('create_reminder',{'activity':'eat prawns','delay':10,'unit':'minutes','allow_duplicate':True})
        result=self.contextualizer.receive_message(Message.from_content('Create an additional eating reminder in ten minutes'))
        self.assertEqual(len(result.tasks),1)
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],2)

    def test_context_update_and_inbox_completion_are_committed_together(self):
        from message_queue import MessageQueue,MessageWorker
        self.controller.run_once()
        inbox=MessageQueue(self.path,'u1')
        message=Message.from_content('I have prawns allergy')
        inbox.enqueue(message)
        self.model.chat.return_value=call('add_task_context',{'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'Thanks for letting me know.'})
        MessageWorker(inbox,self.contextualizer).run_once()
        self.assertEqual(inbox.status(message.id),'DONE')
        with connect(self.path) as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
            self.assertIn('prawns allergy',db.execute("SELECT description FROM actions WHERE status='PENDING'").fetchone()[0])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM notifications WHERE notification_id=?',('context-'+message.id,)).fetchone()[0],1)

    def test_context_is_retained_by_recurring_delivery(self):
        from unittest.mock import patch
        self.controller.run_once()
        with connect(self.path) as db:
            db.execute('UPDATE tasks SET period=86400')
        self.enrich()
        due=datetime.fromisoformat(self.original.actions[0].context['requested_time'])+timedelta(seconds=1)
        with patch('action_queue.utc_now',return_value=due):
            self.executor.execute(self.queue.claim_due())
        with connect(self.path) as db:
            next_action=db.execute("SELECT description,context FROM actions WHERE status='PENDING'").fetchone()
            self.assertIsNone(next_action)
            self.assertEqual(db.execute('SELECT status FROM tasks').fetchone()[0],'ACTIVE')

    def test_context_update_uses_model_reply_instead_of_fixed_status_text(self):
        self.controller.run_once()
        self.model.chat.return_value=call('add_task_context',{
            'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'Thanks for sharing that with me.'})
        self.contextualizer.receive_message(Message.from_content('I have prawns allergy'))
        messages=[row['message'] for row in self.queue.pending_notifications()]
        self.assertEqual(messages,['Thanks for sharing that with me.'])
        with connect(self.path) as db:
            self.assertIn('I have prawns allergy',db.execute("SELECT description FROM actions WHERE status='PENDING'").fetchone()[0])

    def test_stale_allergy_fact_cannot_be_attached_to_a_later_no(self):
        self.controller.run_once()
        self.model.chat.side_effect=[call('add_task_context',{
            'confirm':True,'information':'I have prawns allergy','shared_reference':'prawns','reply':'I have a prawn allergy.'}),
            {'role':'assistant','content':'No problem.'}]
        result=self.contextualizer.receive_message(Message.from_content('no'))
        self.assertEqual(result.reply,'No problem.')
        with connect(self.path) as db:
            row=db.execute("SELECT description,context FROM actions WHERE status='PENDING'").fetchone()
            self.assertNotIn('User-provided context',row['description'])
            self.assertNotIn('Note:',row['context'])
        self.assertFalse(any(row['message']=='I have a prawn allergy.' for row in self.queue.pending_notifications()))

    def test_unrelated_fact_and_feedback_do_not_mutate_actions(self):
        self.controller.run_once()
        with connect(self.path) as db:
            db.execute("UPDATE tasks SET description='eat'")
            db.execute("UPDATE actions SET description='Remind user to eat'")
            before=[tuple(row) for row in db.execute('SELECT * FROM actions ORDER BY action_id')]
        for text, reference in [('I have prawns allergy','prawns'), ('no','no'), ("I don’t get it",'get')]:
            self.model.chat.side_effect=[call('add_task_context',{
                'confirm':True,'information':text,'shared_reference':reference,'reply':'Bad update'}),
                {'role':'assistant','content':'Let’s clarify that.'}]
            result=self.contextualizer.receive_message(Message.from_content(text))
            self.assertEqual(result.reply,'Let’s clarify that.')
            self.assertEqual(self.model.chat.call_args.kwargs['tools'],[])
        with connect(self.path) as db:
            self.assertEqual(before,[tuple(row) for row in db.execute('SELECT * FROM actions ORDER BY action_id')])
            self.assertEqual(db.execute('SELECT COUNT(*) FROM tasks').fetchone()[0],1)
