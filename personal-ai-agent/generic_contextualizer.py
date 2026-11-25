"""Bounded generic planning; native operation batches never execute external work."""
import json
import logging
from datetime import datetime
from pathlib import Path
from time import monotonic

from database import connect
from llm import LLMError, EmptyModelResponse, ModelTimeoutError
from models import ActionContext, Message
from planner_operations import PLANNER_TOOLS, apply_operations, initialize_planner, committed_result
from storage import MessageStore
from task_store import TaskStore

logger = logging.getLogger(__name__)
PROMPT = (Path(__file__).parent/'prompts'/'contextualizer.txt').read_text()


def receive(contextualizer, message, queue_token):
    store = contextualizer.message_store
    store.save(message, user_id=contextualizer.user_id, queue_message_id=message.id if queue_token else None, queue_token=queue_token)
    initialize_planner(store.db_path)
    prior = committed_result(store.db_path,message,contextualizer.user_id)
    if prior is not None:
        return prior
    history = store.full_history(contextualizer.user_id,message.id)
    tasks = TaskStore(store.db_path).user_context(contextualizer.user_id)
    with connect(store.db_path) as db:
        conversational = [dict(row) for row in db.execute("SELECT * FROM actions WHERE task_id IS NULL AND user_id=? AND status IN ('PENDING','RUNNING')",(contextualizer.user_id,))]
    context = tasks + [{'task_id': None, 'actions': conversational}]
    data = {'received_at':message.timestamp,'now':datetime.now(contextualizer.timezone).isoformat(),
            'timezone':contextualizer.timezone.key,'tasks':tasks,'conversational_actions':conversational,
            'recent_dialogue':[{'role':item['role'],'message':item['message'],'timestamp':item['timestamp']} for item in history[-8:]]}
    messages = [{'role':'system','content':PROMPT+'\nCurrent context:\n'+json.dumps(data,ensure_ascii=False)}]
    messages.extend({'role':item['role'],'content':item['message']} for item in history)
    messages.append({'role':'user','content':'CURRENT REQUEST:\n'+message.message})
    deadline = monotonic()+contextualizer.planning_timeout
    for step in range(3):
        remaining = deadline-monotonic()
        if remaining<=0:
            raise ModelTimeoutError('Planning budget exhausted; no changes committed')
        logger.debug('planner_llm_start message=%s step=%s',message.id,step+1)
        try:
            response = contextualizer.llm.chat(messages, tools=PLANNER_TOOLS, timeout=remaining)
        except EmptyModelResponse:
            messages.append({'role':'system','content':'Call create_action exactly once. Contextualizer does not communicate with the user.'})
            continue
        if monotonic()>deadline:
            raise ModelTimeoutError('Planning budget exhausted; no changes committed')
        if not isinstance(response,dict) or response.get('role')!='assistant':
            raise LLMError('Invalid planner response')
        content = response.get('content') or ''
        calls = response.get('tool_calls') or []
        if not isinstance(content,str) or not isinstance(calls,list):
            raise LLMError('Invalid planner response fields')
        if not calls:
            messages.append(response)
            messages.append({'role':'system','content':'Do not reply to the user. Call create_action exactly once with message_type, task_id, description and context, including general conversation.'})
            continue
        if any(not isinstance(call,dict) or not isinstance(call.get('function'),dict) for call in calls):
            raise LLMError('Invalid native planner tool call')
        messages.append({key:value for key,value in response.items() if key in {'role','content','tool_calls'}})
        try:
            operations=[]
            for call in calls:
                function = call['function']
                args = function['arguments']
                if isinstance(args,str):
                    args=json.loads(args)
                operations.append({'name':function['name'],'arguments':args})
            logger.debug('planner_operations message=%s operations=%s',message.id,json.dumps(operations,ensure_ascii=False))
            result = apply_operations(store.db_path,contextualizer.user_id,message,operations,context,
                                      reply=None,queue_token=queue_token)
            logger.info('planner_committed message=%s tools=%s',message.id,','.join(op['name'] for op in operations))
            return result
        except (ValueError,TypeError,KeyError) as error:
            # Claim fencing errors must never be converted into model retries.
            from message_queue import StaleMessageClaim
            if isinstance(error,StaleMessageClaim):
                raise
            logger.info('planner_batch_rejected message=%s reason=%s',message.id,str(error))
            for call in calls:
                result = {'role':'tool','tool_name':call.get('function',{}).get('name','invalid'),
                          'content':json.dumps({'error':str(error),'instruction':'No handoff committed. Return exactly one corrected create_action call.'})}
                if call.get('id'):
                    result['tool_call_id']=call['id']
                messages.append(result)
    raise LLMError('Planner could not produce a valid operation batch; no changes committed')
