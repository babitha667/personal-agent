"""Local-only draft and email mailbox. No SMTP or external delivery."""
import argparse
import hashlib
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from database import connect

EMAIL_FIELDS = {'email_id', 'to', 'subject', 'content', 'thread_id'}


def initialize_email_tables(path: Path) -> None:
    from database import initialize_task_tables
    initialize_task_tables(path)
    with connect(path) as db:
        db.execute('''CREATE TABLE IF NOT EXISTS emails (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            email_id TEXT UNIQUE NOT NULL,
            thread_id TEXT NOT NULL,
            user_id TEXT NOT NULL CHECK(length(trim(user_id))>0),
            timestamp TEXT NOT NULL,
            "from" TEXT NOT NULL,
            "to" TEXT NOT NULL,
            subject TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL CHECK(length(trim(content))>0),
            status TEXT NOT NULL DEFAULT 'SENT' CHECK(status IN ('DRAFT','SENT','RECEIVED')),
            revision INTEGER NOT NULL DEFAULT 0,
            read INTEGER NOT NULL DEFAULT 0 CHECK(read IN (0,1)),
            action_id TEXT UNIQUE REFERENCES actions(action_id))''')
        columns = {row['name'] for row in db.execute('PRAGMA table_info(emails)')}
        if 'read' not in columns:
            db.execute('ALTER TABLE emails ADD COLUMN read INTEGER NOT NULL DEFAULT 0 CHECK(read IN (0,1))')
        if 'subject' not in columns:
            db.execute("ALTER TABLE emails ADD COLUMN subject TEXT NOT NULL DEFAULT ''")
        if 'revision' not in columns:
            db.execute('ALTER TABLE emails ADD COLUMN revision INTEGER NOT NULL DEFAULT 0')
        if 'status' not in columns:
            db.execute("ALTER TABLE emails ADD COLUMN status TEXT NOT NULL DEFAULT 'SENT' CHECK(status IN ('DRAFT','SENT','RECEIVED'))")
            # Existing outgoing records really committed: never relabel them drafts.
            db.execute("UPDATE emails SET status='RECEIVED' WHERE action_id IS NULL")
        db.execute('CREATE INDEX IF NOT EXISTS idx_emails_unread ON emails(user_id,read,timestamp)')
        db.execute('CREATE INDEX IF NOT EXISTS idx_emails_user_thread ON emails(user_id,thread_id,timestamp,sequence)')


def address(value):
    if not isinstance(value, str) or not re.fullmatch(r'[^\s@<>]+@[^\s@<>]+\.[^\s@<>]+', value.strip()):
        raise ValueError('A valid email address is required')
    return value.strip()


def text(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Subject and content must be nonempty text')
    return value.strip()


def mailbox_address(user_id):
    demo = 'user-' + hashlib.sha256(user_id.encode()).hexdigest()[:12] + '@local.invalid'
    return address(os.getenv('AGENT_EMAIL_ADDRESS', demo))


def owned_thread(db, user_id, thread_id):
    if not isinstance(thread_id, str) or not db.execute(
            'SELECT 1 FROM emails WHERE thread_id=? AND user_id=?', (thread_id, user_id)).fetchone():
        raise ValueError('Thread not found in this user mailbox')


def prepare_email(db, user_id, arguments, *, draft=False):
    """Validate and stage creation, draft editing, or draft sending; no writes."""
    if not isinstance(arguments, dict) or set(arguments) - EMAIL_FIELDS:
        raise ValueError('Unexpected email arguments; sender and user_id are backend-owned')
    email_id = arguments.get('email_id')
    existing = None
    if email_id is not None:
        if not isinstance(email_id, str):
            raise ValueError('email_id must be a string')
        existing = db.execute('SELECT * FROM emails WHERE email_id=? AND user_id=?', (email_id, user_id)).fetchone()
        if existing is None:
            raise ValueError('Email not found in this user mailbox')
        if existing['status'] != 'DRAFT':
            raise ValueError('This message is already sent/received and cannot be changed or sent again; only a DRAFT can be edited or sent by email_id')
        if arguments.get('thread_id') not in (None,existing['thread_id']):
            raise ValueError('An existing draft must keep its thread_id')
    recipient_value = arguments.get('to', existing['to'] if existing else None)
    subject_value = arguments.get('subject', existing['subject'] if existing else None)
    recipient = '' if draft and recipient_value in (None,'') else address(recipient_value)
    subject = '' if draft and subject_value in (None,'') else text(subject_value)
    content = text(arguments.get('content', existing['content'] if existing else None))
    thread_id = existing['thread_id'] if existing else arguments.get('thread_id')
    if thread_id is not None:
        owned_thread(db, user_id, thread_id)
    return {
        'email_id': existing['email_id'] if existing else str(uuid4()),
        'thread_id': thread_id or str(uuid4()), 'thread_id_existing': thread_id is not None,
        'from': mailbox_address(user_id), 'to': recipient, 'subject': subject, 'content': content,
        'read': True, 'status': 'DRAFT' if draft else 'SENT',
        'existing': existing is not None,
        'expected_revision': existing['revision'] if existing else None,
    }


def commit_email(db, user_id, action_id, record, timestamp):
    """Commit atomically with fenced action completion; sending a draft changes its state."""
    if record['thread_id_existing']:
        owned_thread(db, user_id, record['thread_id'])
    if record.get('existing'):
        updated = db.execute('''UPDATE emails SET timestamp=?,"from"=?,"to"=?,subject=?,content=?,
            status=?,read=1,revision=revision+1 WHERE email_id=? AND user_id=? AND status='DRAFT' AND revision=?''',
            (timestamp, record['from'], record['to'], record['subject'], record['content'], record['status'],
             record['email_id'], user_id, record['expected_revision']))
        if updated.rowcount != 1:
            raise ValueError('Draft changed or was already sent; reload it before proceeding. No changes committed.')
    else:
        db.execute('''INSERT INTO emails(email_id,thread_id,user_id,timestamp,"from","to",subject,content,status,read,action_id)
            VALUES(?,?,?,?,?,?,?,?,?,?,?)''',
            (record['email_id'], record['thread_id'], user_id, timestamp, record['from'], record['to'],
             record['subject'], record['content'], record['status'], int(record['read']), action_id))
    return {key: record[key] for key in ('email_id', 'thread_id', 'from', 'to', 'subject', 'content', 'status')} | {
        'user_id': user_id, 'timestamp': timestamp, 'read': bool(record['read']), 'local_only': True}


class EmailStore:
    def __init__(self, path: Path, user_id: str):
        self.path, self.user_id = path, user_id
        initialize_email_tables(path)

    def read(self, *, thread_id=None, email_id=None, limit=50, mark_read=False, unread_only=False, status=None):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError('limit must be between 1 and 100')
        if type(mark_read) is not bool or type(unread_only) is not bool:
            raise ValueError('Read options must be boolean')
        if thread_id is not None and email_id is not None:
            raise ValueError('Select thread_id OR email_id')
        if status is not None and status not in ('DRAFT', 'SENT', 'RECEIVED'):
            raise ValueError('Invalid email status filter')
        columns = 'email_id,thread_id,user_id,timestamp,"from","to",subject,content,status,read'
        conditions, parameters = ['user_id=?'], [self.user_id]
        if unread_only:
            conditions.append('read=0')
        if status is not None:
            conditions.append('status=?'); parameters.append(status)
        with connect(self.path) as db:
            if thread_id is not None:
                owned_thread(db, self.user_id, thread_id)
                conditions.append('thread_id=?'); parameters.append(thread_id)
            if email_id is not None:
                conditions.append('email_id=?'); parameters.append(email_id)
            where = ' AND '.join(conditions)
            rows = db.execute(f'''SELECT {columns} FROM (SELECT * FROM emails WHERE {where}
                ORDER BY julianday(timestamp) DESC,sequence DESC LIMIT ?)
                ORDER BY julianday(timestamp),sequence''', (*parameters, limit)).fetchall()
            records = [dict(row, read=bool(row['read'])) for row in rows]
            if mark_read:
                db.executemany('UPDATE emails SET read=1 WHERE email_id=? AND user_id=?',
                               [(r['email_id'], self.user_id) for r in records])
                for record in records:
                    record['read'] = True
        return records

    def receive(self, sender, content, *, thread_id=None, subject=None):
        sender = address(sender)
        content = text(content)
        now = datetime.now(timezone.utc).isoformat()
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            if thread_id is not None:
                owned_thread(db, self.user_id, thread_id)
                previous = db.execute('SELECT subject FROM emails WHERE thread_id=? AND user_id=? ORDER BY sequence DESC LIMIT 1',
                                      (thread_id, self.user_id)).fetchone()
                if subject is None:
                    subject = previous['subject']
            record = {'email_id': str(uuid4()), 'thread_id': thread_id or str(uuid4()),
                      'thread_id_existing': thread_id is not None, 'from': sender,
                      'to': mailbox_address(self.user_id), 'subject': subject or '', 'content': content,
                      'read': False, 'status': 'RECEIVED', 'existing': False}
            return commit_email(db, self.user_id, None, record, now)


def main():
    parser = argparse.ArgumentParser(description='Local demo mailbox; no external email delivery')
    parser.add_argument('--user', default=os.getenv('AGENT_USER_ID', 'u1'))
    commands = parser.add_subparsers(dest='command', required=True)
    read = commands.add_parser('list'); read.add_argument('--thread'); read.add_argument('--status', choices=['DRAFT', 'SENT', 'RECEIVED'])
    receive = commands.add_parser('receive'); receive.add_argument('--thread'); receive.add_argument('--from', dest='sender', required=True)
    receive.add_argument('--content', required=True); receive.add_argument('--subject')
    args = parser.parse_args()
    path = Path(os.getenv('AGENT_DB_PATH', str(Path(__file__).parent / 'data' / 'agent.db'))).resolve()
    store = EmailStore(path, args.user)
    if args.command == 'list':
        for row in store.read(thread_id=args.thread, status=args.status): print(row)
    else:
        print(store.receive(args.sender, args.content, thread_id=args.thread, subject=args.subject))


if __name__ == '__main__': main()
