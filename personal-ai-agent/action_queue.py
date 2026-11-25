"""Durable due-action claims, fenced leases, retries, and atomic completion."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from database import connect, initialize_task_tables
from models import Action
from action_data import action_from_record, parse_context, encode_context, merge_context
from action_history import task_history, snapshot

ACTION_COLUMNS = 'a.action_id, a.task_id, a.exec_at, a.description, a.status, a.attempt_count, a.created_at, a.context'


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def initialize_execution_tables(path: Path) -> None:
    initialize_task_tables(path)
    from outbox import initialize_outbox
    initialize_outbox(path)
    from email_store import initialize_email_tables
    initialize_email_tables(path)
    with connect(path) as connection:
        connection.execute('''CREATE TABLE IF NOT EXISTS action_leases (
            action_id TEXT PRIMARY KEY REFERENCES actions(action_id),
            token TEXT NOT NULL, lease_until TEXT NOT NULL)''')
        connection.execute('''CREATE TABLE IF NOT EXISTS notifications (
            notification_id TEXT PRIMARY KEY, action_id TEXT NOT NULL REFERENCES actions(action_id),
            user_id TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL,
            delivered_at TEXT)''')
        connection.execute('''CREATE TABLE IF NOT EXISTS action_results (
            action_id TEXT PRIMARY KEY REFERENCES actions(action_id), outcome TEXT NOT NULL,
            reason TEXT NOT NULL, completed_at TEXT NOT NULL, details TEXT NOT NULL DEFAULT '{}')''')
        result_columns = {row[1] for row in connection.execute('PRAGMA table_info(action_results)')}
        if 'details' not in result_columns:
            connection.execute("ALTER TABLE action_results ADD COLUMN details TEXT NOT NULL DEFAULT '{}'")
        connection.execute('CREATE INDEX IF NOT EXISTS idx_actions_due ON actions(status, exec_at)')


@dataclass(frozen=True)
class Claim:
    action: Action
    token: str


class StaleClaim(ValueError):
    """Action was cancelled, reclaimed, or its lease expired."""


class ActionQueue:
    def __init__(self, path: Path, user_id: str = 'u1', lease_seconds: int = 600,
                 max_attempts: int = 3) -> None:
        self.path, self.user_id = path, user_id
        self.lease_seconds, self.max_attempts = lease_seconds, max_attempts
        initialize_execution_tables(path)

    def claim_due(self, now: datetime | None = None) -> Claim | None:
        now = now or utc_now()
        timestamp = now.isoformat()
        with connect(self.path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('''UPDATE actions SET status='PENDING' WHERE status='RUNNING'
                AND (task_id IN (SELECT task_id FROM tasks WHERE user_id=? AND (status='ACTIVE' OR json_extract(actions.context,'$.origin') IN ('contextualizer','action_computer'))) OR (task_id IS NULL AND user_id=?))
                AND NOT EXISTS (SELECT 1 FROM action_leases l WHERE l.action_id=actions.action_id)''', (self.user_id, self.user_id))
            # Only this user's expired claims are recovered. The token fences off old workers.
            connection.execute('''UPDATE actions SET status = 'PENDING' WHERE action_id IN (
                SELECT a.action_id FROM actions a JOIN action_leases l USING(action_id)
                LEFT JOIN tasks t ON t.task_id = a.task_id WHERE COALESCE(t.user_id,a.user_id) = ? AND (a.task_id IS NULL OR t.status = 'ACTIVE' OR json_extract(a.context,'$.origin') IN ('contextualizer','action_computer'))
                AND a.status = 'RUNNING' AND julianday(l.lease_until) <= julianday(?))''',
                (self.user_id, timestamp))
            self._fail_exhausted_conversations(connection)
            exhausted = [row[0] for row in connection.execute('''SELECT DISTINCT a.task_id
                FROM actions a JOIN tasks t ON t.task_id = a.task_id
                WHERE t.user_id = ? AND t.status = 'ACTIVE' AND a.status = 'PENDING'
                AND a.attempt_count >= ? AND NOT COALESCE((json_extract(a.context,'$.origin') IN ('contextualizer','action_computer') AND COALESCE(json_extract(a.context,'$.message_type'),'') NOT IN ('new_task','recurring_occurrence')),0)''', (self.user_id, self.max_attempts))]
            for task_id in exhausted:
                connection.execute("UPDATE actions SET status = 'FAILED' WHERE task_id = ? AND status = 'PENDING' AND attempt_count >= ?",
                                   (task_id, self.max_attempts))
                self._fail_task(connection, task_id)
            connection.execute('''UPDATE actions SET status = 'CANCELLED' WHERE status IN ('PENDING', 'RUNNING')
                AND COALESCE(json_extract(context,'$.origin'),'') NOT IN ('contextualizer','action_computer') AND task_id IN (SELECT task_id FROM tasks WHERE user_id = ? AND status != 'ACTIVE')''', (self.user_id,))
            connection.execute('''DELETE FROM action_leases WHERE action_id IN (
                SELECT a.action_id FROM actions a LEFT JOIN tasks t ON t.task_id=a.task_id
                WHERE COALESCE(t.user_id,a.user_id)=? AND a.status != 'RUNNING')''', (self.user_id,))
            row = connection.execute(f'''SELECT {ACTION_COLUMNS} FROM actions a
                LEFT JOIN tasks t ON t.task_id = a.task_id
                WHERE COALESCE(t.user_id,a.user_id) = ? AND (a.task_id IS NULL OR t.status = 'ACTIVE' OR json_extract(a.context,'$.origin') IN ('contextualizer','action_computer')) AND a.status = 'PENDING'
                AND a.attempt_count < ? AND julianday(a.exec_at) <= julianday(?)
                ORDER BY julianday(a.exec_at), a.action_id LIMIT 1''', (self.user_id, self.max_attempts, timestamp)).fetchone()
            if row is None:
                return None
            action = dict(row)
            action['status'] = 'RUNNING'
            action['attempt_count'] += 1
            token = str(uuid4())
            connection.execute("UPDATE actions SET status='RUNNING', attempt_count=attempt_count+1 WHERE action_id=?", (action['action_id'],))
            connection.execute('INSERT OR REPLACE INTO action_leases VALUES (?, ?, ?)',
                               (action['action_id'], token, (now + timedelta(seconds=self.lease_seconds)).isoformat()))
            return Claim(action_from_record(action), token)

    def context(self, claim: Claim, now: datetime | None = None) -> dict:
        with connect(self.path) as connection:
            self._check(connection, claim, now or utc_now())
            source = None
            if connection.execute("SELECT 1 FROM sqlite_master WHERE name='action_sources'").fetchone():
                source_row = connection.execute('SELECT m.timestamp,m.message FROM action_sources s JOIN messages m ON m.id=s.message_id WHERE s.action_id=?',(claim.action.action_id,)).fetchone()
                source = dict(source_row) if source_row else None
            phase = 'KICKOFF' if source else 'CONTINUATION'
            if claim.action.task_id is None:
                return {'task_id': None, 'user_id': self.user_id, 'actions': [], 'source_message': source, 'phase': phase}
            task = dict(connection.execute('SELECT * FROM tasks WHERE task_id=?', (claim.action.task_id,)).fetchone())
            task['actions'] = [{key:row[key] for key in row.keys() if key!='context'} for row in connection.execute(
                f'SELECT {ACTION_COLUMNS} FROM actions a WHERE task_id=? ORDER BY created_at, action_id', (claim.action.task_id,))]
            task['phase'] = phase
            task['source_message'] = source
            result = connection.execute('''SELECT r.* FROM action_results r JOIN actions a USING(action_id)
                WHERE a.task_id=? ORDER BY r.completed_at DESC LIMIT 1''', (claim.action.task_id,)).fetchone()
            task['last_result'] = dict(result,details=parse_context(result['details'])) if result else None
            return task

    def _check(self, connection, claim: Claim, now: datetime) -> None:
        row = connection.execute('''SELECT 1 FROM actions a JOIN action_leases l USING(action_id)
            LEFT JOIN tasks t ON t.task_id=a.task_id WHERE a.action_id=? AND l.token=?
            AND a.status='RUNNING' AND (a.task_id IS NULL OR t.status='ACTIVE' OR json_extract(a.context,'$.origin') IN ('contextualizer','action_computer')) AND COALESCE(t.user_id,a.user_id)=?
            AND julianday(l.lease_until)>julianday(?)''',
            (claim.action.action_id, claim.token, self.user_id, now.isoformat())).fetchone()
        if row is None:
            raise StaleClaim('Action cancelled, lease expired, or action already handled')

    @staticmethod
    def _fail_task(connection, task_id: str) -> None:
        connection.execute("UPDATE tasks SET status='FAILED' WHERE task_id=? AND status='ACTIVE'", (task_id,))
        connection.execute("UPDATE actions SET status='CANCELLED' WHERE task_id=? AND status IN ('PENDING', 'RUNNING') AND COALESCE(json_extract(context,'$.origin'),'') NOT IN ('contextualizer','action_computer')", (task_id,))
        failed = connection.execute('''SELECT a.action_id, t.user_id, t.description FROM actions a
            JOIN tasks t ON t.task_id=a.task_id WHERE a.task_id=? AND a.status='FAILED'
            ORDER BY a.created_at LIMIT 1''', (task_id,)).fetchone()
        if failed:
            connection.execute('''INSERT OR IGNORE INTO notifications
                (notification_id, action_id, user_id, message, created_at)
                VALUES (?, ?, ?, ?, ?)''',
                (f"failure-{failed['action_id']}", failed['action_id'], failed['user_id'],
                 f"I couldn't complete this task: {failed['description']}. Please check the log and retry or update it.", utc_now().isoformat()))

    def _fail_exhausted_conversations(self, connection) -> None:
        rows = connection.execute("""SELECT a.action_id FROM actions a LEFT JOIN tasks t USING(task_id)
            WHERE COALESCE(t.user_id,a.user_id)=? AND (a.task_id IS NULL OR (json_extract(a.context,'$.origin') IN ('contextualizer','action_computer') AND COALESCE(json_extract(a.context,'$.message_type'),'') NOT IN ('new_task','recurring_occurrence')))
            AND a.status IN ('PENDING','FAILED') AND a.attempt_count>=?""", (self.user_id,self.max_attempts)).fetchall()
        for row in rows:
            connection.execute("UPDATE actions SET status='FAILED' WHERE action_id=?", (row['action_id'],))
            connection.execute("INSERT OR IGNORE INTO notifications VALUES (?,?,?,?,?,NULL)",
                ('failure-'+row['action_id'],row['action_id'],self.user_id,"I couldn't process that message. Please try again.",utc_now().isoformat()))

    def retry(self, claim: Claim, now: datetime | None = None) -> None:
        now = now or utc_now()
        with connect(self.path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            # A stale/cancelled worker must never revive old work.
            try:
                self._check(connection, claim, now)
            except StaleClaim:
                return
            if claim.action.attempt_count >= self.max_attempts:
                connection.execute("UPDATE actions SET status='FAILED' WHERE action_id=?", (claim.action.action_id,))
                is_administrative=(claim.action.context.get('origin') in ('contextualizer','action_computer') and claim.action.context.get('message_type') not in ('new_task','recurring_occurrence'))
                if claim.action.task_id is None or is_administrative:
                    self._fail_exhausted_conversations(connection)
                else:
                    self._fail_task(connection, claim.action.task_id)
            else:
                retry_at = (now + timedelta(seconds=5 * 2 ** (claim.action.attempt_count - 1))).isoformat()
                connection.execute("UPDATE actions SET status='PENDING', exec_at=? WHERE action_id=?", (retry_at, claim.action.action_id))
            connection.execute('DELETE FROM action_leases WHERE action_id=? AND token=?', (claim.action.action_id, claim.token))

    def finish(self, claim: Claim, outcome: str, follow_ups: list[dict],
               notification: str | None, now: datetime | None = None, *, reason: str = '', task_status: str | None = None, email: dict | None = None, read_email_ids: list[str] | None = None, management=None) -> list[str]:
        """Commit effects, follow-ups, and action/task transitions exactly once per claim."""
        now = now or utc_now()
        timestamp = now.isoformat()
        new_ids = []
        with connect(self.path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            self._check(connection, claim, now)
            if management is not None:
                task_id=management.commit(connection)
                claim=replace(claim,action=replace(claim.action,task_id=task_id))
            task = connection.execute('SELECT * FROM tasks WHERE task_id=?', (claim.action.task_id,)).fetchone()

            if task_status is not None:
                if task is None or task_status not in ('ACTIVE','COMPLETED','FAILED'):
                    raise ValueError('Invalid task status update')
                if outcome == 'needs_input' and task_status != 'ACTIVE':
                    raise ValueError('Waiting tasks must remain ACTIVE')
                if (outcome == 'failed') != (task_status == 'FAILED'):
                    raise ValueError('Task status and failure outcome disagree')
            for email_id in read_email_ids or []:
                updated=connection.execute('UPDATE emails SET read=1 WHERE email_id=? AND user_id=?',(email_id,self.user_id))
                if updated.rowcount != 1:
                    raise ValueError('Cannot mark another user email read')
            email_result = None
            if email is not None:
                if claim.action.task_id is None or outcome != 'completed':
                    raise ValueError('Sending email requires a task and a completed action')
                from email_store import commit_email
                email_result = commit_email(connection,self.user_id,claim.action.action_id,email,timestamp)
            current_action = dict(connection.execute('SELECT * FROM actions WHERE action_id=?', (claim.action.action_id,)).fetchone())
            if claim.action.task_id is None and follow_ups:
                raise ValueError('Taskless conversational actions cannot create follow-ups')
            if notification:
                connection.execute('INSERT INTO notifications VALUES (?, ?, ?, ?, ?, NULL)',
                                   (claim.action.action_id, claim.action.action_id, self.user_id, notification, timestamp))
            successors = [dict(item) for item in follow_ups]
            for successor in successors:
                if successor.get('description') == claim.action.description:
                    successor['description'] = current_action['description']
            managed_recurrence = bool(task is not None and task['period'])
            new_ids = [str(uuid4()) for _ in successors]
            result = {'outcome': {'completed':'COMPLETED','skipped':'SKIPPED','needs_input':'WAITING','failed':'FAILED'}[outcome],
                      'reason':reason,'completed_at':timestamp,
                      'details':{'message':notification,'follow_up_action_ids':new_ids}}
            if management is not None and management.changed:
                result['details']['management']={'updated_task_ids':sorted(management.touched_tasks),
                    'created_task_ids':sorted(management.new_tasks),
                    'invalidated_action_ids':sorted(management.touched_actions),
                    'replacement_action_ids':sorted(management.new_actions)}
            if read_email_ids:
                result['details']['read_email']={'marked_read_email_ids':list(read_email_ids)}
            if email_result is not None:
                result['details']['draft_email' if email_result['status']=='DRAFT' else 'send_email'] = email_result
            history = task_history(connection,claim.action.task_id)
            completed_current = dict(current_action,status='FAILED' if outcome=='failed' else 'COMPLETED')
            history = [entry for entry in history if entry['action_id']!=claim.action.action_id]
            history.append(snapshot(completed_current,result))
            if management is not None:
                for replacement_id in management.new_actions:
                    row=connection.execute('SELECT context FROM actions WHERE action_id=?',(replacement_id,)).fetchone()
                    facts=parse_context(row['context']); facts['past_actions']=history
                    connection.execute('UPDATE actions SET context=? WHERE action_id=?',(encode_context(facts),replacement_id))
            for action_id, follow_up in zip(new_ids,successors):
                context = merge_context(current_action['context'],follow_up.get('context', {}))
                context['past_actions'] = history
                context.pop('origin',None)
                context.pop('message_type',None)
                connection.execute('''INSERT INTO actions
                    (action_id, task_id, exec_at, description, status, attempt_count, created_at, context)
                    VALUES (?, ?, ?, ?, 'PENDING', 0, ?, ?)''',
                    (action_id, claim.action.task_id, follow_up['exec_at'], follow_up['description'], timestamp,
                     encode_context(context)))
            connection.execute('INSERT INTO action_results(action_id,outcome,reason,completed_at,details) VALUES(?,?,?,?,?)',
                               (claim.action.action_id,result['outcome'],reason,timestamp,encode_context(result['details'])))
            connection.execute('UPDATE actions SET status=? WHERE action_id=?',
                               ('FAILED' if outcome == 'failed' else 'COMPLETED', claim.action.action_id))
            connection.execute('DELETE FROM action_leases WHERE action_id=?', (claim.action.action_id,))
            if task is None:
                return new_ids
            if task['status'] != 'ACTIVE':
                if task_status is not None and task_status != task['status']:
                    raise ValueError('Cannot change terminal task state through finish_action')
                return new_ids
            outstanding = connection.execute("SELECT 1 FROM actions WHERE task_id=? AND status IN ('PENDING','RUNNING') AND COALESCE(json_extract(context,'$.origin'),'') NOT IN ('contextualizer','action_computer') LIMIT 1",(claim.action.task_id,)).fetchone()
            if task_status=='COMPLETED' and (managed_recurrence or outstanding):
                raise ValueError('Cannot complete a task while future work remains')
            if outcome=='failed':
                self._fail_task(connection,claim.action.task_id)
            elif task_status is not None:
                connection.execute('UPDATE tasks SET status=? WHERE task_id=?',(task_status,claim.action.task_id))
            elif management is None and not outstanding and outcome!='needs_input' and not managed_recurrence:
                # Compatibility for direct callers. Native Executor must explicitly assert task completion.
                connection.execute("UPDATE tasks SET status='COMPLETED' WHERE task_id=?",(claim.action.task_id,))
            # Advance recurrence only after executing an occurrence, never after metadata work.
            occurrence = managed_recurrence and claim.action.context.get('message_type') in (None,'recurring_occurrence') and not (management and management.changed)
            if outcome in ('completed','skipped') and not outstanding and (task_status=='COMPLETED' or occurrence or (management is None and not managed_recurrence)):
                connection.execute('UPDATE tasks SET last_executed_at=? WHERE task_id=?',(timestamp,claim.action.task_id))
        return new_ids

    def pending_notifications(self) -> list[dict]:
        with connect(self.path) as connection:
            return [dict(row) for row in connection.execute('''SELECT * FROM notifications
                WHERE user_id=? AND delivered_at IS NULL ORDER BY created_at, notification_id''', (self.user_id,))]

    def mark_delivered(self, notification_id: str) -> None:
        from storage import MessageStore
        MessageStore(self.path)  # Ensure conversation schema exists for standalone databases.
        with connect(self.path) as connection:
            connection.execute('BEGIN IMMEDIATE')
            notification = connection.execute('SELECT * FROM notifications WHERE notification_id=? AND user_id=?',
                                              (notification_id, self.user_id)).fetchone()
            if notification is None:
                return
            if not notification_id.startswith('ack-'):
                connection.execute("INSERT OR IGNORE INTO messages (id,timestamp,message,user_id,role,kind) VALUES (?,?,?,?,'assistant',?)",
                                   ('notification-' + notification_id, notification['created_at'], notification['message'], self.user_id,
                                    'notification' if notification['action_id'] else 'chat'))
            connection.execute('UPDATE notifications SET delivered_at=? WHERE notification_id=? AND user_id=?',
                               (utc_now().isoformat(), notification_id, self.user_id))
