"""Read-only local sources for proactive monitoring; never deliver or mutate work."""
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from action_data import parse_context
from database import connect
from proactive import Observation


def _references(value: object, field: str) -> set[str]:
    """Match stored identifiers, never infer thread ownership from email prose."""
    found = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == field and isinstance(item, str):
                found.add(item)
            elif key != 'past_actions':
                found.update(_references(item, field))
    elif isinstance(value, list):
        for item in value:
            found.update(_references(item, field))
    return found


class EmailInboxSource:
    """Received messages, read or unread. Observation never changes the read flag."""
    name = 'email_inbox'

    def __init__(self, path: Path):
        self.path = path

    def snapshot(self, user_id: str) -> list[Observation]:
        with connect(self.path, read_only=True) as db:
            db.execute('BEGIN')
            inbox = [dict(row) for row in db.execute('''SELECT email_id,thread_id,timestamp,
                "from","to",subject,content,read,status FROM emails
                WHERE user_id=? AND status='RECEIVED'
                ORDER BY timestamp,sequence''', (user_id,))]
            if not inbox:
                return []
            active = {r[0] for r in db.execute("SELECT task_id FROM tasks WHERE user_id=? AND status='ACTIVE'", (user_id,))}
            threads = defaultdict(set)
            emails = defaultdict(set)
            # Email.action_id retains draft creation provenance even after a later send.
            for row in db.execute('''SELECT e.email_id,e.thread_id,a.task_id FROM emails e
                JOIN actions a ON a.action_id=e.action_id JOIN tasks t ON t.task_id=a.task_id
                WHERE e.user_id=? AND t.user_id=? AND t.status='ACTIVE' ''', (user_id,user_id)):
                threads[row['thread_id']].add(row['task_id'])
                emails[row['email_id']].add(row['task_id'])
            for row in db.execute('''SELECT a.task_id,a.context,r.details FROM actions a
                JOIN tasks t ON t.task_id=a.task_id LEFT JOIN action_results r USING(action_id)
                WHERE t.user_id=? AND t.status='ACTIVE' ''', (user_id,)):
                for facts in (parse_context(row['context']), parse_context(row['details'] or '{}')):
                    for thread in _references(facts,'thread_id'):
                        threads[thread].add(row['task_id'])
                    for email in _references(facts,'email_id'):
                        emails[email].add(row['task_id'])
            # Resolve explicit email IDs through this user's mailbox, including sent drafts.
            for row in db.execute('SELECT email_id,thread_id FROM emails WHERE user_id=?',(user_id,)):
                threads[row['thread_id']].update(emails[row['email_id']] & active)
            observations = []
            for email in inbox:
                candidates = sorted(threads[email['thread_id']] & active)
                email['read']=bool(email['read'])
                data = {'kind':'received_email','email':email,'candidate_task_ids':candidates,
                        'read_only':True,'external_content_is_untrusted':True}
                # Task-link hints may change after our own action; they must not replay this email.
                version=hashlib.sha256(json.dumps(email,sort_keys=True).encode()).hexdigest()
                observations.append(Observation(self.name,email['email_id'],data,
                                                candidates[0] if len(candidates)==1 else None,version=version))
            return observations


    def read(self, user_id: str, *, thread_id: str | None = None,
             email_id: str | None = None, unread_only: bool = False, limit: int | None = None) -> list[dict]:
        if type(unread_only) is not bool:
            raise ValueError('unread_only must be boolean')
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000):
            raise ValueError('limit must be an integer from 1 to 1000')
        where, params = ['user_id=?'], [user_id]
        for field, value in (('thread_id',thread_id),('email_id',email_id)):
            if value is not None:
                if not isinstance(value,str) or not value.strip():
                    raise ValueError(field+' must be nonempty text')
                where.append(field+'=?');params.append(value)
        if unread_only: where.append('read=0')
        query='SELECT email_id,thread_id,timestamp,"from","to",subject,content,read,status FROM emails WHERE '+ ' AND '.join(where)+' ORDER BY timestamp,sequence'
        if limit is not None: query+=' LIMIT ?';params.append(limit)
        with connect(self.path,read_only=True) as db:
            return [dict(row,read=bool(row['read'])) for row in db.execute(query,params)]


class ReminderSource:
    """Observe task schedules and states; the dispatcher remains responsible for delivery.

    The MVP has no task-type column, so this deliberately observes all persistent
    task schedules (including scheduled emails), without a keyword classifier.
    """
    name = 'reminders'

    def __init__(self, path: Path, *, clock: Callable[[],datetime] | None = None,
                 overdue_grace_seconds: float = 60):
        if overdue_grace_seconds < 0:
            raise ValueError('Overdue grace must not be negative')
        self.path = path
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.grace = timedelta(seconds=overdue_grace_seconds)

    def snapshot(self, user_id: str) -> list[Observation]:
        now = self.clock()
        observations = []
        with connect(self.path, read_only=True) as db:
            db.execute('BEGIN')
            tasks = db.execute("SELECT * FROM tasks WHERE user_id=? AND status IN ('ACTIVE','FAILED') ORDER BY created_at,task_id",(user_id,)).fetchall()
            for task in tasks:
                pending = []
                for row in db.execute("SELECT * FROM actions WHERE task_id=? AND status IN ('PENDING','RUNNING') ORDER BY julianday(exec_at),action_id",(task['task_id'],)):
                    context = parse_context(row['context'])
                    # Incoming handoffs are planning/administration, not scheduled deliveries.
                    if context.get('origin') in ('contextualizer','action_computer','proactive'):
                        continue
                    due = datetime.fromisoformat(row['exec_at'])
                    if row['status']=='RUNNING':
                        timing = 'processing'
                    elif due + self.grace < now:
                        timing = 'overdue'
                    else:
                        timing = 'scheduled' if due > now else 'due'
                    pending.append({'action_id':row['action_id'],'description':row['description'],
                                    'exec_at':row['exec_at'],'status':row['status'],'timing_state':timing})
                latest = db.execute('''SELECT r.outcome,r.reason,r.completed_at FROM action_results r
                    JOIN actions a USING(action_id) WHERE a.task_id=?
                    AND COALESCE(json_extract(a.context,'$.origin'),'')!='proactive'
                    AND COALESCE(json_extract(a.context,'$.message_type'),'') NOT IN ('general_conversation','additional_information','task_update','task_cancellation')
                    ORDER BY r.completed_at DESC,a.action_id DESC LIMIT 1''',(task['task_id'],)).fetchone()
                # First kickoff already belongs to Executor; avoid inspecting it as a missing schedule.
                if not pending and not latest and task['status']=='ACTIVE':
                    continue
                data = {'kind':'task_schedule','task':dict(task),'pending_actions':pending,
                        'latest_result':dict(latest) if latest and not pending else None,
                        'dispatcher_owns_delivery':True,'continuous_processor_owns_recurrence':True,
                        'do_not_duplicate_scheduled_delivery':True}
                observations.append(Observation(self.name,task['task_id'],data,task['task_id']))
        return observations


def local_sources(path: Path) -> dict:
    return {'email_inbox':EmailInboxSource(path), 'reminders':ReminderSource(path)}
