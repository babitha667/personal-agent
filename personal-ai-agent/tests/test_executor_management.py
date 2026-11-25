"""Executor-owned management, atomic effects, timing and conversational references."""
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
from continuous_user_processor import ContinuousUserProcessor
from database import connect
from executor_tools import ExecutorTools
from models import Message
from storage import MessageStore
from task_store import TaskStore
from test_generic_planner import batch,create,handoff


class ExecutorManagementTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.path=Path(temp.name)/'agent.db'
        self.model=Mock();self.contextualizer=Contextualizer(MessageStore(self.path),llm=self.model)
        self.queue=ActionQueue(self.path)
        self.executor_model=Mock();self.executor=ActionExecutor(self.queue,llm=self.executor_model)

    def incoming(self,text,response):
        self.model.chat.return_value=response
        return self.contextualizer.receive_message(Message.from_content(text))

    def pending(self,task_id):
        return [a for a in TaskStore(self.path).user_context('u1') if a['task_id']==task_id][0]['actions']

    def schedule(self,period=None):
        result=self.incoming('Remind me about my important meeting in ten minutes',
            create('Attend important meeting','Arrange the meeting reminder',context={'constraints':{'meeting':'important'}},period=period))
        self.executor_model.chat.return_value=batch(
            ('add_actions',{'actions':[{'exec_at':{'delay_seconds':600},'description':'Remind the user to attend the meeting'}]}),
            ('send_message',{'message':'I’ll remind you in ten minutes.'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Scheduling committed','task_status':'ACTIVE'}))
        self.executor.execute(self.queue.claim_due())
        delivery=next(a for a in self.pending(result.tasks[0].task_id) if a['status']=='PENDING')
        return result,delivery

    def update(self,result,text,kind='additional_information'):
        self.incoming(text,handoff(result.tasks[0].task_id,'Evaluate the current request and update affected work',
            context={'user_requirement':text},message_type=kind))
        return ExecutorTools(self.queue,self.queue.claim_due())

    def finish(self,tools,message='Noted.',status='ACTIVE'):
        tools.execute('send_message',{'message':message})
        args={'outcome':'COMPLETED','reason':'Verified management committed'}
        if status:args['task_status']=status
        return tools.execute('finish_action',args)

    def test_general_chat_is_executed_without_persistent_task(self):
        self.incoming('Hey!',handoff())
        self.assertEqual(self.queue.pending_notifications(),[])
        self.executor_model.chat.return_value=batch(('send_message',{'message':'Hey! How can I help?'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Answered'}))
        self.executor.execute(self.queue.claim_due())
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])
        self.assertEqual([n['message'] for n in self.queue.pending_notifications()],['Hey! How can I help?'])

    def test_scheduling_is_not_delivery_and_task_stays_active(self):
        result,delivery=self.schedule()
        due=datetime.fromisoformat(result.incoming_message.timestamp)+timedelta(minutes=10)
        self.assertEqual(datetime.fromisoformat(delivery['exec_at']),due)
        self.assertIsNone(self.queue.claim_due(now=due-timedelta(seconds=1)))
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'ACTIVE')
        self.assertEqual(len(self.queue.pending_notifications()),1)
        self.executor_model.chat.return_value=batch(('send_message',{'message':'Time to attend your meeting.'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Delivered','task_status':'COMPLETED'}))
        with patch('action_queue.utc_now',return_value=due):self.executor.execute(self.queue.claim_due())
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'COMPLETED')

    def test_meeting_notes_are_merged_semantically_without_shared_phrase(self):
        result,delivery=self.schedule()
        text='Great, this meeting is very improtant to me, I need to take my notes to this meeting'
        tools=self.update(result,text)
        replacement=tools.execute('update_action',{'action_id':delivery['action_id'],
            'description':'Remind the user to attend the meeting and take their notes',
            'context':{'bring':['notes'],'importance':'very important'}})
        # Until finish, original task/action and outbox are untouched.
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'PENDING')
        self.finish(tools,'Got it — make sure you have your notes with you.')
        updated=ActionStore(self.path).get(replacement['replacement_action_id'])
        self.assertEqual(updated.exec_at,delivery['exec_at'])
        self.assertEqual(updated.context['bring'],['notes'])
        self.assertEqual(updated.context['constraints'],{'meeting':'important'})
        self.assertEqual(updated.context['past_actions'][-1]['outcome'],'COMPLETED')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'CANCELLED')
        self.assertEqual(len(TaskStore(self.path).user_context('u1')),1)
        self.executor_model.chat.return_value=batch(('send_message',{'message':'Your meeting is starting — bring your notes.'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Delivered with updated context','task_status':'COMPLETED'}))
        due=datetime.fromisoformat(updated.exec_at)
        with patch('action_queue.utc_now',return_value=due):self.executor.execute(self.queue.claim_due())
        self.assertIn('notes',json.dumps(self.executor_model.chat.call_args.args[0]))

    def test_rescheduling_replaces_old_pending_action_without_duplicate_task(self):
        result,delivery=self.schedule()
        tools=self.update(result,'Make that two minutes from now','task_update')
        request=tools.claim.action.context['request'];due=datetime.fromisoformat(request['received_at'])+timedelta(minutes=2)
        tools.execute('update_task',{'description':'Attend important meeting in two minutes'})
        replaced=tools.execute('update_action',{'action_id':delivery['action_id'],'exec_at':due.isoformat()})
        self.finish(tools,'Moved it to two minutes from now.')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'CANCELLED')
        pending=[a for a in self.pending(result.tasks[0].task_id) if a['status']=='PENDING']
        self.assertEqual(len(pending),1)
        self.assertEqual(pending[0]['action_id'],replaced['replacement_action_id'])
        self.assertEqual(datetime.fromisoformat(pending[0]['exec_at']),due)
        self.assertEqual(len(TaskStore(self.path).user_context('u1')),1)

    def test_cancel_then_add_one_next_followup_is_atomic(self):
        result,delivery=self.schedule();tools=self.update(result,'In three minutes instead','task_update')
        tools.execute('cancel_action',{'action_id':delivery['action_id']})
        tools.execute('add_actions',{'actions':[{'exec_at':{'delay_seconds':180},'description':'Remind the user to attend the meeting'}]})
        self.finish(tools,'Rescheduled.')
        pending=[a for a in self.pending(result.tasks[0].task_id) if a['status']=='PENDING']
        self.assertEqual(len(pending),1)
        self.assertEqual(datetime.fromisoformat(pending[0]['exec_at']),datetime.fromisoformat(tools.claim.action.context['request']['received_at'])+timedelta(minutes=3))

    def test_task_cancellation_invalidates_work_and_keeps_terminal_state(self):
        result,delivery=self.schedule();tools=self.update(result,'Cancel that meeting reminder','task_cancellation')
        tools.execute('cancel_task',{})
        self.finish(tools,'Cancelled.',status=None)
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'CANCELLED')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'CANCELLED')
        self.assertFalse(any(a['status'] in ('PENDING','RUNNING') for a in self.pending(result.tasks[0].task_id)))

    def test_new_information_does_not_restart_recurrence_clock(self):
        result,delivery=self.schedule(period=86400)
        tools=self.update(result,'Bring notes to this meeting')
        tools.execute('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}})
        self.finish(tools)
        self.assertIsNone(TaskStore(self.path).get(result.tasks[0].task_id).last_executed_at)

    def test_recurring_occurrences_belong_to_continuous_processor(self):
        result,delivery=self.schedule(period=60);due=datetime.fromisoformat(delivery['exec_at'])
        self.executor_model.chat.return_value=batch(('send_message',{'message':'Time for the meeting.'}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Occurrence delivered','task_status':'ACTIVE'}))
        with patch('action_queue.utc_now',return_value=due):self.executor.execute(self.queue.claim_due())
        task=TaskStore(self.path).get(result.tasks[0].task_id)
        self.assertEqual(task.status,'ACTIVE');self.assertEqual(datetime.fromisoformat(task.last_executed_at),due)
        self.assertFalse(any(a['status']=='PENDING' for a in self.pending(task.task_id)))
        processor=ContinuousUserProcessor(self.path,agent=Mock());processor.register_user('u1')
        self.assertEqual(processor.trigger_recurring('u1',due+timedelta(seconds=59)),0)
        self.assertEqual(processor.trigger_recurring('u1',due+timedelta(seconds=60)),1)
        self.assertEqual(processor.trigger_recurring('u1',due+timedelta(seconds=61)),0)

    def test_waiting_for_details_keeps_task_active(self):
        result=self.incoming('Remind me about the meeting',create('Meeting'))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('send_message',{'message':'When should I remind you?'})
        tools.execute('finish_action',{'outcome':'WAITING','reason':'Need a time','task_status':'ACTIVE'})
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'ACTIVE')
        response=self.incoming('In five minutes',handoff(result.tasks[0].task_id,
            'Continue the reminder using the clarified time',message_type='task_response'))
        self.assertEqual(response.tasks,[])
        self.assertEqual(response.actions[0].task_id,result.tasks[0].task_id)

    def test_cannot_complete_task_while_delivery_is_pending(self):
        result,delivery=self.schedule();tools=self.update(result,'Thanks','task_response')
        tools.execute('send_message',{'message':'You’re welcome.'})
        with self.assertRaises(ValueError):tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Chat answered','task_status':'COMPLETED'})
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'ACTIVE')
        self.assertEqual(len(self.queue.pending_notifications()),1)

    def test_failed_batch_rolls_back_task_action_changes_and_message(self):
        result,delivery=self.schedule();self.update(result,'Bring notes to the meeting')
        # Release the staged fixture claim to execute through the real bounded tool driver.
        with connect(self.path) as db:
            db.execute("UPDATE actions SET status='PENDING' WHERE status='RUNNING'")
            db.execute('DELETE FROM action_leases')
        invalid=batch(('update_task',{'description':'Changed'}),('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}}),
            ('send_message',{'message':'Updated successfully.'}),('unknown_tool',{}),
            ('finish_action',{'outcome':'COMPLETED','reason':'Not valid'}))
        corrected=batch(('send_message',{'message':'Could you clarify?'}),('finish_action',{'outcome':'WAITING','reason':'Need clarification','task_status':'ACTIVE'}))
        self.executor_model.chat.side_effect=[invalid,corrected]
        self.executor.execute(self.queue.claim_due())
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).description,'Attend important meeting')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'PENDING')
        self.assertFalse(any(n['message']=='Updated successfully.' for n in self.queue.pending_notifications()))

    def test_concurrent_change_rejects_stale_update_and_success_message(self):
        result,delivery=self.schedule();tools=self.update(result,'Bring notes')
        tools.execute('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}})
        with connect(self.path) as db:db.execute('UPDATE actions SET description=? WHERE action_id=?',('Changed elsewhere',delivery['action_id']))
        with self.assertRaisesRegex(ValueError,'changed'):self.finish(tools,'Updated successfully.')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).description,'Changed elsewhere')
        self.assertEqual(len(self.queue.pending_notifications()),1)

    def test_replaced_running_delivery_is_fenced(self):
        result,delivery=self.schedule()
        # Claim delivery with a fixture clock but execute the immediate update before that worker commits.
        due=datetime.fromisoformat(delivery['exec_at'])
        old_claim=self.queue.claim_due(now=due)
        tools=self.update(result,'Bring notes')
        tools.execute('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}})
        self.finish(tools)
        with self.assertRaises(StaleClaim):self.queue.finish(old_claim,'completed',[],'Obsolete reminder')

    def test_ownership_and_current_action_mutations_are_rejected(self):
        result,delivery=self.schedule();tools=self.update(result,'Bring notes')
        for action_id in ('someone-elses-action',tools.claim.action.action_id):
            with self.assertRaises(ValueError):tools.execute('cancel_action',{'action_id':action_id})
        with self.assertRaises(ValueError):tools.execute('select_task',{'task_id':'someone-elses-task'})
        tasks=tools.execute('read_tasks',{})['tasks']
        self.assertEqual({t['user_id'] for t in tasks},{'u1'})

    def test_sequential_native_tools_have_room_to_finish(self):
        result,delivery=self.schedule();self.update(result,'Bring notes')
        with connect(self.path) as db:
            db.execute("UPDATE actions SET status='PENDING' WHERE status='RUNNING'");db.execute('DELETE FROM action_leases')
        self.executor_model.chat.side_effect=[
            batch(('read_actions',{})),batch(('update_task',{'description':'Attend meeting with notes'})),
            batch(('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}})),
            batch(('send_message',{'message':'I’ll include your notes.'})),
            batch(('finish_action',{'outcome':'COMPLETED','reason':'Updated','task_status':'ACTIVE'}))]
        self.executor.execute(self.queue.claim_due())
        self.assertEqual(self.executor_model.chat.call_count,6) # Includes initial scheduling call.
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).description,'Attend meeting with notes')

    def test_failed_administrative_action_does_not_cancel_valid_reminder(self):
        result,delivery=self.schedule()
        self.incoming('Bring my notes',handoff(result.tasks[0].task_id,'Evaluate notes',message_type='additional_information'))
        now=datetime.now(timezone.utc)+timedelta(seconds=1)
        for attempt in range(3):
            claim=self.queue.claim_due(now=now)
            self.assertEqual(claim.action.context['message_type'],'additional_information')
            self.queue.retry(claim,now=now)
            now+=timedelta(seconds=30)
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'ACTIVE')
        self.assertEqual(ActionStore(self.path).get(delivery['action_id']).status,'PENDING')
        self.assertIsNone(self.queue.claim_due(now=now))
        self.assertTrue(any("couldn't process" in n['message'] for n in self.queue.pending_notifications()))

    def test_administrative_chat_does_not_implicitly_complete_waiting_task(self):
        result=self.incoming('Help me arrange something',create('Arrange an appointment'))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('send_message',{'message':'Which day works for you?'})
        tools.execute('finish_action',{'outcome':'WAITING','reason':'Need a day','task_status':'ACTIVE'})
        self.incoming('How is this progressing?',handoff(result.tasks[0].task_id,'Explain that a day is still needed'))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('send_message',{'message':'I’m waiting for your preferred day.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Answered the status question'})
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'ACTIVE')
        self.assertIsNone(TaskStore(self.path).get(result.tasks[0].task_id).last_executed_at)

    def test_taskless_context_can_select_owned_task_and_stage_changes(self):
        result,delivery=self.schedule()
        self.incoming('Add my notes to that request',handoff(description='Resolve the related task and include notes',message_type='task_response'))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        tools.execute('select_task',{'task_id':result.tasks[0].task_id})
        tools.execute('update_action',{'action_id':delivery['action_id'],'context':{'bring':['notes']}})
        self.finish(tools)
        with connect(self.path) as db:
            current=db.execute('SELECT task_id FROM actions WHERE action_id=?',(tools.claim.action.action_id,)).fetchone()[0]
            self.assertEqual(current,result.tasks[0].task_id)
        self.assertEqual(len(TaskStore(self.path).user_context('u1')),1)

    def test_task_creation_for_unbound_request_is_staged_and_generic_chat_is_guarded(self):
        self.incoming('An unbound clarified request',handoff(message_type='task_response'))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        created=tools.execute('create_task',{'description':'Clarified user request'})
        self.assertEqual(TaskStore(self.path).user_context('u1'),[])
        tools.execute('send_message',{'message':'Which date should I use?'})
        tools.execute('finish_action',{'outcome':'WAITING','reason':'Date missing','task_status':'ACTIVE'})
        self.assertEqual(TaskStore(self.path).get(created['task_id']).status,'ACTIVE')
        self.incoming('Hey!',handoff())
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        with self.assertRaises(ValueError):tools.execute('create_task',{'description':'Greeting'})

    def test_queued_input_can_inspect_task_that_completed_before_dispatch(self):
        result,delivery=self.schedule()
        self.incoming('Was that already delivered?',handoff(result.tasks[0].task_id,'Check delivery status'))
        with connect(self.path) as db:
            db.execute("UPDATE tasks SET status='COMPLETED' WHERE task_id=?",(result.tasks[0].task_id,))
            db.execute("UPDATE actions SET status='COMPLETED' WHERE action_id=?",(delivery['action_id'],))
        claim=self.queue.claim_due();self.assertIsNotNone(claim)
        tools=ExecutorTools(self.queue,claim)
        tools.execute('send_message',{'message':'That reminder was already delivered.'})
        tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Status answered'})
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'COMPLETED')

    def test_replacement_of_pending_kickoff_preserves_original_receipt(self):
        initial=self.incoming('Remind me in five minutes',create('Meeting'))
        self.incoming('Take notes',handoff(initial.tasks[0].task_id,'Add notes to the pending request',message_type='additional_information'))
        # Claim the input update first with explicit ordering in this concurrency fixture.
        with connect(self.path) as db:
            db.execute('UPDATE actions SET exec_at=? WHERE action_id=?',((datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),initial.actions[0].action_id))
        tools=ExecutorTools(self.queue,self.queue.claim_due())
        replaced=tools.execute('update_action',{'action_id':initial.actions[0].action_id,'context':{'bring':['notes']}})
        self.finish(tools)
        due=datetime.fromisoformat(ActionStore(self.path).get(replaced['replacement_action_id']).exec_at)
        with patch('action_queue.utc_now',return_value=due):
            claim=self.queue.claim_due()
            self.assertEqual(self.queue.context(claim)['source_message']['timestamp'],initial.incoming_message.timestamp)
            next_tools=ExecutorTools(self.queue,claim)
            next_tools.execute('add_actions',{'actions':[{'exec_at':{'delay_seconds':300},'description':'Remind about the meeting and notes'}]})
        expected=datetime.fromisoformat(initial.incoming_message.timestamp)+timedelta(minutes=5)
        self.assertEqual(datetime.fromisoformat(next_tools.follow_ups[0]['exec_at']),expected)

    def test_completion_only_turn_is_reserved_and_bounded(self):
        self.incoming('Hello',handoff())
        self.executor_model.chat.side_effect=[batch(('read_tasks',{})) for _ in range(6)]+[
            batch(('finish_action',{'outcome':'SKIPPED','reason':'No further work appropriate'}))]
        self.executor.execute(self.queue.claim_due())
        self.assertEqual(self.executor_model.chat.call_count,7)
        final_tools=self.executor_model.chat.call_args.kwargs['tools']
        self.assertEqual([t['function']['name'] for t in final_tools],['finish_action'])

    def test_completion_ignores_queued_administrative_handoffs_without_losing_them(self):
        result,delivery=self.schedule()
        due=datetime.fromisoformat(delivery['exec_at'])
        delivery_claim=self.queue.claim_due(now=due)
        self.incoming('Was that sent?',handoff(result.tasks[0].task_id,'Answer delivery status'))
        with patch('action_queue.utc_now',return_value=due):
            tools=ExecutorTools(self.queue,delivery_claim)
            tools.execute('send_message',{'message':'Time for your meeting.'})
            tools.execute('finish_action',{'outcome':'COMPLETED','reason':'Delivered','task_status':'COMPLETED'})
            question=self.queue.claim_due()
            self.assertIsNotNone(question)
            self.assertEqual(question.action.context['message_type'],'general_conversation')
        self.assertEqual(TaskStore(self.path).get(result.tasks[0].task_id).status,'COMPLETED')
