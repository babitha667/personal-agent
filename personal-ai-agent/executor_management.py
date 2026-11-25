"""User-scoped, staged task/action management committed with executor effects.

Reads use an in-memory working view. Original rows are compared under the queue's
write transaction so concurrent changes cannot be overwritten or reported successful.
"""
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from uuid import uuid4

from action_data import action_record, context_input, merge_context, parse_context
from action_history import with_history
from database import connect
from models import Action, Task
from planning import nonempty_text
from scheduling import resolve_execution_time
from tool_definitions import definition

TEXT={'type':'string','minLength':1}
CONTEXT={'type':'object'}
MANAGEMENT_TOOLS=[
    definition('read_tasks','Read this user\'s tasks, states and actions. Other users are inaccessible.',{},[]),
    definition('select_task','Associate this execution with an existing owned task when context resolves an unbound request. Cannot switch an already associated task.',{'task_id':TEXT},['task_id']),
    definition('create_task','Stage a task only if this action has no associated task and a persistent user request requires one. Contextualizer normally created the initial task. Never duplicate tasks or create one for casual chat.',
        {'description':TEXT,'period':{'type':['integer','null'],'minimum':1}},['description']),
    definition('update_task','Stage changes to the associated task description or recurrence period. Does not imply completion.',
        {'description':TEXT,'period':{'type':['integer','null'],'minimum':1}},[]),
    definition('cancel_task','Cancel the associated task and invalidate its other pending/running actions atomically at finish_action.',{},[]),
    definition('update_action','Replace an outstanding action on the associated task, fencing any worker. Omitted exec_at preserves timing; context merges with existing facts. Cannot modify the current action.',
        {'action_id':TEXT,'description':TEXT,'context':CONTEXT,
         'exec_at':{'type':'string','description':'Timezone-aware execution timestamp; omit to preserve due time'}},['action_id']),
    definition('cancel_action','Invalidate one outstanding action on the associated task. Cannot cancel the current action; finish it instead.',{'action_id':TEXT},['action_id']),
]
MANAGEMENT_NAMES={t['function']['name'] for t in MANAGEMENT_TOOLS}


class ManagementConflict(ValueError):
    """Concurrent state changed; abandon staging and retry with a fresh snapshot."""


class ExecutorManagement:
    def __init__(self,queue,claim):
        self.user_id=queue.user_id
        self.current_id=claim.action.action_id
        self.task_id=claim.action.task_id
        self.original_tasks={}
        self.original_actions={}
        with connect(queue.path) as db:
            for row in db.execute('SELECT * FROM tasks WHERE user_id=?',(self.user_id,)):
                self.original_tasks[row['task_id']]=dict(row)
            for row in db.execute('''SELECT a.* FROM actions a LEFT JOIN tasks t USING(task_id)
                WHERE COALESCE(t.user_id,a.user_id)=?''',(self.user_id,)):
                self.original_actions[row['action_id']]=dict(row)
        self.tasks=deepcopy(self.original_tasks)
        self.actions=deepcopy(self.original_actions)
        self.touched_tasks=set()
        self.touched_actions=set()
        self.new_actions=set()
        self.new_tasks=set()
        self.cancelled=False
        self.message_type=claim.action.context.get('message_type')
        self.replacements={}

    @property
    def changed(self):
        return bool(self.touched_tasks or self.touched_actions or self.new_tasks or self.new_actions)

    def _task(self):
        row=self.tasks.get(self.task_id)
        if row is None: raise ValueError('Select or create an associated task first')
        if row['status']!='ACTIVE': raise ValueError('Task is no longer ACTIVE; inspect latest state before continuing')
        return row

    @staticmethod
    def _period(value):
        if value is not None and (type(value) is not int or not 0<value<=2**63-1):
            raise ValueError('Period must be a positive integer in seconds or null')

    def read(self):
        return [dict(row,actions=[dict(a,context=parse_context(a['context'])) for a in self.actions.values()
                                 if a['task_id']==row['task_id']]) for row in self.tasks.values()]

    def execute(self,name,args,now):
        if not isinstance(args,dict): raise ValueError('Arguments must be an object')
        schema=next(t['function']['parameters'] for t in MANAGEMENT_TOOLS if t['function']['name']==name)
        if set(args)-set(schema['properties']) or set(schema['required'])-set(args):
            raise ValueError('Unexpected or missing management fields')
        if name=='read_tasks': return {'tasks':self.read()}
        if name=='select_task':
            task_id=nonempty_text(args['task_id'])
            if self.task_id is not None and task_id!=self.task_id: raise ValueError('Cannot switch associated task')
            if task_id not in self.tasks: raise ValueError('Task is not owned by this user')
            self.task_id=task_id
            return {'selected_task_id':task_id,'task':self.tasks[task_id]}
        if name=='create_task':
            if self.message_type=='general_conversation': raise ValueError('General conversation cannot create a persistent task')
            if self.task_id is not None: raise ValueError('An associated task already exists; update it instead')
            description=nonempty_text(args['description']);period=args.get('period');self._period(period)
            if any(t['status']=='ACTIVE' and t['description'].strip().casefold()==description.casefold() for t in self.tasks.values()):
                raise ValueError('Identical active task exists; select it instead')
            task=Task(str(uuid4()),self.user_id,description,'ACTIVE',period,now.isoformat())
            self.task_id=task.task_id
            self.tasks[task.task_id]=asdict(task);self.new_tasks.add(task.task_id)
            return {'staged':True,'task_id':task.task_id}
        task=self._task()
        if name=='update_task':
            if not args: raise ValueError('Supply changed task fields')
            changes={}
            if 'description' in args: changes['description']=nonempty_text(args['description'])
            if 'period' in args: self._period(args['period']);changes['period']=args['period']
            task.update(changes)
            self.touched_tasks.add(self.task_id)
            return {'staged':True,'task':task}
        if name=='cancel_task':
            task['status']='CANCELLED';self.touched_tasks.add(self.task_id);self.cancelled=True
            for action in self.actions.values():
                if action['task_id']==self.task_id and action['action_id']!=self.current_id and action['status'] in ('PENDING','RUNNING') and parse_context(action['context']).get('origin') not in ('contextualizer','action_computer'):
                    self.touched_actions.add(action['action_id']);action['status']='CANCELLED'
            return {'staged':True,'task_id':self.task_id,'status':'CANCELLED'}
        action_id=nonempty_text(args['action_id'])
        action=self.actions.get(action_id)
        if action_id==self.current_id or action is None or action['task_id']!=self.task_id or action['status'] not in ('PENDING','RUNNING'):
            raise ValueError('Select another outstanding action on the associated owned task')
        if name=='cancel_action':
            action['status']='CANCELLED';self.touched_actions.add(action_id)
            return {'staged':True,'cancelled_action_id':action_id}
        if set(args)=={'action_id'}: raise ValueError('Supply changed action fields')
        facts=merge_context(action['context'],context_input(args.get('context',{})))
        due=action['exec_at']
        if 'exec_at' in args:
            if not isinstance(args['exec_at'],str): raise ValueError('exec_at must be an aware timestamp')
            due=resolve_execution_time(args['exec_at'],now,now).isoformat()
        facts.pop('origin',None)
        facts.pop('message_type',None)
        replacement=Action(str(uuid4()),self.task_id,due,nonempty_text(args.get('description',action['description'])),
                           'PENDING',0,now.isoformat(),context=facts)
        action['status']='CANCELLED';self.touched_actions.add(action_id)
        self.actions[replacement.action_id]=dict(action_record(replacement),user_id=self.user_id)
        self.new_actions.add(replacement.action_id)
        self.replacements[replacement.action_id]=action_id
        return {'staged':True,'cancelled_action_id':action_id,'replacement_action_id':replacement.action_id,'exec_at':due}

    def commit(self,db):
        """Called only inside ActionQueue.finish's fenced BEGIN IMMEDIATE."""
        # Compare the associated task even for send-only effects: cancellation/state races invalidate work.
        checked_tasks=self.touched_tasks | ({self.task_id} if self.task_id in self.original_tasks else set())
        for task_id in checked_tasks:
            if task_id in self.new_tasks: continue
            row=db.execute('SELECT * FROM tasks WHERE task_id=? AND user_id=?',(task_id,self.user_id)).fetchone()
            if row is None or dict(row)!=self.original_tasks[task_id]:
                raise ManagementConflict('Task changed during execution; retry with fresh state')
        for action_id in self.touched_actions:
            if action_id in self.new_actions: continue
            row=db.execute('SELECT * FROM actions WHERE action_id=?',(action_id,)).fetchone()
            if row is None or dict(row)!=self.original_actions[action_id]:
                raise ManagementConflict('Pending action changed during execution; retry with fresh state')
        for task_id in self.new_tasks:
            task=self.tasks[task_id]
            if db.execute("SELECT 1 FROM tasks WHERE user_id=? AND status='ACTIVE' AND lower(trim(description))=lower(trim(?))",(self.user_id,task['description'])).fetchone():
                raise ManagementConflict('Concurrent duplicate task; retry with fresh state')
            db.execute('INSERT INTO tasks VALUES(:task_id,:user_id,:description,:status,:period,:created_at,:last_executed_at)',task)
        for task_id in self.touched_tasks:
            task=self.tasks[task_id]
            db.execute('UPDATE tasks SET description=?,period=?,status=? WHERE task_id=? AND user_id=?',
                       (task['description'],task['period'],task['status'],task_id,self.user_id))
        for action_id in self.touched_actions:
            db.execute("UPDATE actions SET status='CANCELLED' WHERE action_id=?",(action_id,))
            db.execute('DELETE FROM action_leases WHERE action_id=?',(action_id,))
        # Cancellation must also catch actions enqueued since the read snapshot.
        if self.cancelled:
            db.execute("UPDATE actions SET status='CANCELLED' WHERE task_id=? AND action_id!=? AND status IN ('PENDING','RUNNING') AND COALESCE(json_extract(context,'$.origin'),'') NOT IN ('contextualizer','action_computer')",(self.task_id,self.current_id))
            db.execute("DELETE FROM action_leases WHERE action_id IN (SELECT action_id FROM actions WHERE task_id=? AND status='CANCELLED')",(self.task_id,))
        for action_id in self.new_actions:
            record=dict(self.actions[action_id])
            facts=with_history(db,self.task_id,parse_context(record['context']))
            record['context']=json.dumps(facts)
            db.execute('''INSERT INTO actions(action_id,task_id,exec_at,description,status,attempt_count,created_at,user_id,context)
                VALUES(:action_id,:task_id,:exec_at,:description,:status,:attempt_count,:created_at,:user_id,:context)''',record)
            # Preserve kickoff receipt anchoring, without fabricating a source for deliveries.
            parent=self.replacements[action_id]
            while parent in self.replacements: parent=self.replacements[parent]
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='action_sources'").fetchone():
                source=db.execute('SELECT message_id FROM action_sources WHERE action_id=?',(parent,)).fetchone()
                if source: db.execute('INSERT INTO action_sources VALUES(?,?)',(action_id,source['message_id']))
        if self.task_id!=self.original_actions[self.current_id]['task_id']:
            db.execute('UPDATE actions SET task_id=? WHERE action_id=?',(self.task_id,self.current_id))
        return self.task_id
