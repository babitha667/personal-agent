"""Atomic, idempotent enrichment of active action context."""
import json
from datetime import datetime, timezone
from pathlib import Path

from database import connect
from message_queue import require_claim
from models import Message
from outbox import enqueue_outgoing


class ContextUpdateRejected(ValueError):
    """The proposed context lacks current-message provenance or direct task evidence."""


def add_task_context(path: Path, user_id: str, task_id: str, message: Message, reply: str, reference: str,
                     queue_token: str | None = None) -> int:
    """Preserve the source message verbatim; do not infer or rewrite personal facts."""
    if not isinstance(reference,str) or len(reference.strip()) < 3 or reference.strip().casefold() not in message.message.casefold():
        raise ContextUpdateRejected('Context requires a specific shared reference in the CURRENT message; feedback is not a task fact')
    reference = reference.strip().casefold()
    event_id = 'context-' + message.id
    now = datetime.now(timezone.utc).isoformat()
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        if queue_token is not None:
            require_claim(db, message.id, queue_token)
        if db.execute('SELECT 1 FROM notifications WHERE notification_id=? AND user_id=?',
                      (event_id, user_id)).fetchone():
            return 0
        task = db.execute("SELECT status,description FROM tasks WHERE task_id=? AND user_id=?",
                          (task_id, user_id)).fetchone()
        if task is None or task['status'] != 'ACTIVE':
            raise ValueError('Context can only be added to an ACTIVE task; this task is no longer active')
        action_text = ' '.join(row['description'] for row in db.execute(
            "SELECT description FROM actions WHERE task_id=? AND status IN ('PENDING','RUNNING')", (task_id,)))
        if reference not in (task['description'] + ' ' + action_text).casefold():
            raise ContextUpdateRejected('No direct shared reference with the active action. Leave it unchanged and respond conversationally; a broad topic association is insufficient')
        info = message.message.strip()
        rows=db.execute("SELECT action_id,context FROM actions WHERE task_id=? AND status IN ('PENDING','RUNNING')",(task_id,)).fetchall()
        if not rows:
            raise ValueError('This task has no outstanding action to enrich')
        for row in rows:
            context=json.loads(row['context'])
            facts=context.setdefault('user_information',[])
            if info not in facts: facts.append(info)
            db.execute('UPDATE actions SET description=description || ?,context=? WHERE action_id=?',
                       ('\nUser-provided context: '+info,json.dumps(context),row['action_id']))
        confirmation = reply
        enqueue_outgoing(db, event_id, user_id, confirmation, timestamp=now)
        db.execute('''INSERT OR IGNORE INTO messages
            (id,timestamp,message,user_id,role,kind) VALUES(?,?,?,?,'assistant','plan')''',
            ('notification-' + event_id, now, confirmation, user_id))
        if queue_token is not None:
            db.execute("UPDATE message_inbox SET status='DONE',token=NULL,lease_until=NULL WHERE message_id=? AND token=?",
                       (message.id, queue_token))
        return len(rows)
