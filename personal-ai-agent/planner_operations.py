"""Contextualizer's single durable handoff; initial task creation only, no existing-record mutations."""
import json
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from action_data import action_from_saved, action_record, context_input
from action_history import with_history
from database import connect
from message_queue import require_claim
from models import Action, ActionContext, Task
from planning import nonempty_text
from tool_definitions import definition

MESSAGE_TYPES = ['general_conversation','new_task','additional_information','task_update',
                 'task_cancellation','task_response','recurring_occurrence']
PLANNER_TOOLS = [definition('create_action',
    'Enqueue exactly one immediate handoff for Executor. Identify an owned task when applicable; '
    'For a new request supply new_task to create its initial record atomically. Null task_id is allowed for conversation. Never mutate existing tasks or actions.',
    {'task_id':{'type':['string','null']},'description':{'type':'string','minLength':1},
     'context':{'type':'object','description':'Relevant history, user facts, constraints and intent; structured JSON.'},
     'new_task':{'type':'object','additionalProperties':False,'required':['description'],
                 'properties':{'description':{'type':'string','minLength':1},
                               'period':{'type':['integer','null'],'minimum':1},
                               'allow_duplicate':{'type':'boolean'}}},
     'message_type':{'type':'string','enum':MESSAGE_TYPES}},
    ['task_id','description','context','message_type'])]


def initialize_planner(path):
    with connect(path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS action_sources (action_id TEXT PRIMARY KEY REFERENCES actions(action_id), message_id TEXT NOT NULL REFERENCES messages(id))')
        db.execute('CREATE TABLE IF NOT EXISTS planner_commits (message_id TEXT PRIMARY KEY REFERENCES messages(id), user_id TEXT NOT NULL, result TEXT NOT NULL)')


def committed_result(path, message, user_id):
    with connect(path) as db:
        row=db.execute('SELECT result FROM planner_commits WHERE message_id=? AND user_id=?',(message.id,user_id)).fetchone()
    if row is None: return None
    data=json.loads(row['result'])
    return ActionContext(message,[],[Task(**t) for t in data['tasks']],
                         [action_from_saved(a) for a in data['actions']],data['cancelled_task_ids'],reply=data.get('reply'))


def apply_operations(path, user_id, message, operations, context=None, *, reply=None, queue_token=None):
    """Persist the handoff, receipt and inbox completion in one fenced transaction."""
    if not isinstance(operations,list) or len(operations)!=1:
        raise ValueError('Enqueue exactly one create_action; Contextualizer cannot manage tasks or existing actions')
    operation=operations[0]
    if not isinstance(operation,dict): raise ValueError('Operation must be an object')
    args=operation.get('arguments')
    if operation.get('name')!='create_action' or not isinstance(args,dict):
        raise ValueError('Contextualizer only has create_action')
    required={'task_id','description','context','message_type'}
    if not required <= set(args) or set(args)-required-{'new_task'} or args['message_type'] not in MESSAGE_TYPES:
        raise ValueError('Supply task_id, description, structured context and a valid message_type')
    task_id=args['task_id']
    if task_id is not None: task_id=nonempty_text(task_id)
    description=nonempty_text(args['description'])
    facts=context_input(args['context'])
    now=datetime.now(timezone.utc).isoformat()
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        if queue_token is not None: require_claim(db,message.id,queue_token)
        if db.execute('SELECT 1 FROM planner_commits WHERE message_id=?',(message.id,)).fetchone():
            raise ValueError('Message already committed')
        if task_id is not None and not db.execute('SELECT 1 FROM tasks WHERE task_id=? AND user_id=?',(task_id,user_id)).fetchone():
            raise ValueError('Select an existing owned task, or null for a new request')
        tasks=[]
        if args['message_type']=='new_task':
            proposal=args.get('new_task')
            if task_id is not None or not isinstance(proposal,dict) or set(proposal)-{'description','period','allow_duplicate'} or 'description' not in proposal:
                raise ValueError('New requests require new_task with description and task_id=null')
            description_task=nonempty_text(proposal['description'])
            period=proposal.get('period')
            if period is not None and (type(period) is not int or not 0<period<=2**63-1):
                raise ValueError('Period must be positive integer seconds or null')
            allow=proposal.get('allow_duplicate',False)
            if type(allow) is not bool: raise ValueError('allow_duplicate must be boolean')
            if not allow and db.execute("SELECT 1 FROM tasks WHERE user_id=? AND status='ACTIVE' AND lower(trim(description))=lower(trim(?))",(user_id,description_task)).fetchone():
                raise ValueError('Identical active task exists; identify it instead of duplicating it')
            task=Task(str(uuid4()),user_id,description_task,'ACTIVE',period,now)
            db.execute('INSERT INTO tasks VALUES(:task_id,:user_id,:description,:status,:period,:created_at,:last_executed_at)',asdict(task))
            task_id=task.task_id
            tasks=[task]
        elif 'new_task' in args:
            raise ValueError('Only new_task messages may create an initial task record')
        facts=with_history(db,task_id,facts)
        facts['origin']='contextualizer'
        facts['message_type']=args['message_type']
        facts['request']={'message':message.message,'received_at':message.timestamp}
        action=Action(str(uuid4()),task_id,now,description,'PENDING',0,now,context=facts)
        db.execute('''INSERT INTO actions(action_id,task_id,exec_at,description,status,attempt_count,created_at,user_id,context)
            VALUES(:action_id,:task_id,:exec_at,:description,:status,:attempt_count,:created_at,:user_id,:context)''',
            dict(action_record(action),user_id=user_id))
        db.execute('INSERT INTO action_sources VALUES(?,?)',(action.action_id,message.id))
        result={'tasks':[asdict(t) for t in tasks],'actions':[asdict(action)],'cancelled_task_ids':[],'reply':None}
        db.execute('INSERT INTO planner_commits VALUES(?,?,?)',(message.id,user_id,json.dumps(result)))
        if queue_token is not None:
            db.execute("UPDATE message_inbox SET status='DONE',token=NULL,lease_until=NULL WHERE message_id=? AND token=?",(message.id,queue_token))
    return ActionContext(message,[],tasks=tasks,actions=[action])
