"""Scripted native executor responses for legacy planner regression fixtures."""
import json
from datetime import datetime
from unittest.mock import Mock
from test_generic_planner import batch

def executor_model():
    def respond(messages, **kwargs):
        payload=json.loads(messages[1]['content'])
        action=payload['claimed_action']; context=action['context']
        requested=context.get('requested_time')
        if requested and datetime.fromisoformat(action['exec_at']) < datetime.fromisoformat(requested):
            return batch(('add_actions',{'actions':[{'exec_at':requested,'description':action['description']}]}),
                ('send_message',{'message':'Scheduled the reminder.'}),
                ('finish_action',{'outcome':'COMPLETED','reason':'Scheduled','task_status':'ACTIVE'}))
        text=context.get('legacy_message',action['description'])
        if context.get('user_information'):text+='\n'+ '\n'.join(context['user_information'])
        return batch(('send_message',{'message':text}),('finish_action',{'outcome':'COMPLETED','reason':'Delivered','task_status':'ACTIVE' if payload['task'].get('period') else 'COMPLETED'}))
    return Mock(chat=Mock(side_effect=respond))
