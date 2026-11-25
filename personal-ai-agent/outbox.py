"""One durable outgoing buffer for chat replies, acknowledgments, and action messages."""
from datetime import datetime,timezone
from database import connect

SCHEMA = '''CREATE TABLE {name} (
    notification_id TEXT PRIMARY KEY, action_id TEXT REFERENCES actions(action_id),
    user_id TEXT NOT NULL, message TEXT NOT NULL, created_at TEXT NOT NULL, delivered_at TEXT)'''


def initialize_outbox(path):
    with connect(path) as db:
        db.execute('BEGIN IMMEDIATE')
        exists=db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='notifications'").fetchone()
        if not exists:
            db.execute(SCHEMA.format(name='notifications'))
        elif any(row[1]=='action_id' and row[3] for row in db.execute('PRAGMA table_info(notifications)')):
            db.execute(SCHEMA.format(name='notifications_upgrade'))
            db.execute('INSERT INTO notifications_upgrade SELECT * FROM notifications')
            db.execute('DROP TABLE notifications')
            db.execute('ALTER TABLE notifications_upgrade RENAME TO notifications')


def enqueue_outgoing(db, message_id, user_id, message, *, action_id=None, timestamp=None):
    db.execute('''INSERT OR IGNORE INTO notifications
        (notification_id,action_id,user_id,message,created_at,delivered_at) VALUES(?,?,?,?,?,NULL)''',
        (message_id,action_id,user_id,message,timestamp or datetime.now(timezone.utc).isoformat()))
