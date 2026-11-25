"""Durable FIFO inbox; one message is planned at a time for each user."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread, Lock
from uuid import uuid4

from database import connect
from llm import RetryableLLMError
from models import Message

if TYPE_CHECKING:
    from contextualizer import Contextualizer

logger = logging.getLogger(__name__)


class StaleMessageClaim(ValueError):
    """Another worker reclaimed or completed this message."""


def initialize_inbox(path: Path) -> None:
    with connect(path) as db:
        db.execute('''CREATE TABLE IF NOT EXISTS message_inbox (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE NOT NULL, user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL, message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING' CHECK(status IN ('PENDING','RUNNING','DONE','FAILED')),
            attempts INTEGER NOT NULL DEFAULT 0, token TEXT, lease_until TEXT,
            available_at TEXT NOT NULL, error TEXT)''')


def enqueue_message(db, message: Message, user_id: str, *, acknowledge: bool = True) -> None:
    """Shared insertion for terminal input and transactional internal task triggers."""
    db.execute('INSERT INTO message_inbox(message_id,user_id,timestamp,message,available_at) VALUES(?,?,?,?,?)',
               (message.id,user_id,message.timestamp,message.message,message.timestamp))
    if acknowledge:
        from outbox import enqueue_outgoing
        enqueue_outgoing(db,'ack-'+message.id,user_id,'👍 ACK',timestamp=message.timestamp)


class MessageQueue:
    def __init__(self, path: Path, user_id: str, lease_seconds: float = 180) -> None:
        if not user_id.strip() or lease_seconds <= 0:
            raise ValueError("A user ID and positive message lease are required")
        self.submission_lock = Lock()
        self.path, self.user_id, self.lease_seconds = path, user_id, lease_seconds
        from database import initialize_task_tables
        from outbox import initialize_outbox
        initialize_task_tables(path)
        initialize_outbox(path)
        initialize_inbox(path)

    def enqueue(self, message: Message) -> None:
        with connect(self.path) as db:
            enqueue_message(db,message,self.user_id)

    def claim(self) -> tuple[Message, str] | None:
        now = datetime.now(timezone.utc)
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT * FROM message_inbox WHERE user_id=? AND status IN ('PENDING','RUNNING') ORDER BY sequence LIMIT 1",
                             (self.user_id,)).fetchone()
            if row is None:
                return None
            if row['status'] == 'RUNNING' and row['lease_until'] and datetime.fromisoformat(row['lease_until']) > now:
                return None
            if datetime.fromisoformat(row['available_at']) > now:
                return None
            if row['attempts'] >= 3:
                from outbox import enqueue_outgoing
                error = 'Repeatedly interrupted message processing exceeded its retry limit; the message remains saved.'
                db.execute("UPDATE message_inbox SET status='FAILED',token=NULL,lease_until=NULL,error=? WHERE message_id=?",
                           (error,row['message_id']))
                enqueue_outgoing(db,'error-interrupted-'+row['message_id'],self.user_id,error)
                return None
            token = str(uuid4())
            db.execute("UPDATE message_inbox SET status='RUNNING',attempts=attempts+1,token=?,lease_until=? WHERE message_id=?",
                       (token, (now + timedelta(seconds=self.lease_seconds)).isoformat(), row['message_id']))
            return Message(row['message_id'], row['timestamp'], row['message']), token

    def status(self, message_id: str) -> str | None:
        with connect(self.path) as db:
            row = db.execute('SELECT status FROM message_inbox WHERE message_id=? AND user_id=?',
                             (message_id, self.user_id)).fetchone()
            return row['status'] if row else None

    def finish(self, message_id: str, token: str) -> None:
        with connect(self.path) as db:
            db.execute("UPDATE message_inbox SET status='DONE',token=NULL,lease_until=NULL WHERE message_id=? AND token=? AND status='RUNNING'",
                       (message_id, token))

    def fail(self, message_id: str, token: str, error: str, retry: bool) -> bool:
        from outbox import enqueue_outgoing
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute("SELECT attempts FROM message_inbox WHERE message_id=? AND token=? AND status='RUNNING'",
                             (message_id, token)).fetchone()
            if row is None:
                return False
            retry = retry and row['attempts'] < 3
            db.execute('''UPDATE message_inbox SET status=?,token=NULL,lease_until=NULL,error=?,available_at=?
                          WHERE message_id=? AND token=?''',
                       ('PENDING' if retry else 'FAILED', error,
                        (datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(), message_id, token))
            text = ('Your message is queued for retry. ' if retry else
                    'Your message could not be processed; it remains saved. ') + error
            enqueue_outgoing(db, f'error-{message_id}-{row["attempts"]}', self.user_id, text)
            return retry


class MessageWorker:
    def __init__(self, queue: MessageQueue, contextualizer: Contextualizer) -> None:
        self.queue, self.contextualizer = queue, contextualizer
        self.stop_event = Event()
        self.thread = None

    def run_once(self) -> bool:
        with self.queue.submission_lock:
            claimed = self.queue.claim()
        if claimed is None:
            return False
        message, token = claimed
        try:
            self.contextualizer.receive_message(message, queue_token=token)
            self.queue.finish(message.id, token)
        except StaleMessageClaim:
            logger.info('Stale inbox worker stopped: %s', message.id)
        except Exception as error:
            logger.exception('Message processing failed: %s', message.id)
            if self.queue.status(message.id) == 'DONE':
                self._report(message.id, 'Your request was committed, but saving its response failed. Check the log.')
                return True
            # Network/model availability may recover. Invalid plans are not blindly retried.
            self.queue.fail(message.id, token, str(error), isinstance(error, RetryableLLMError))

        return True

    def _report(self, message_id: str, text: str) -> None:
        from outbox import enqueue_outgoing
        with connect(self.queue.path) as db:
            enqueue_outgoing(db, 'error-'+message_id+'-'+str(datetime.now(timezone.utc).timestamp()), self.queue.user_id, text)

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                worked = self.run_once()
            except Exception:
                logger.exception('Inbox worker iteration failed')
                worked = False
            if not worked:
                self.stop_event.wait(0.25)

    def start(self) -> None:
        self.thread = Thread(target=self.run, name='message-worker', daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)


def require_claim(db, message_id: str, token: str | None) -> None:
    """Fence expired/reclaimed workers before any conversation/plan write."""
    row = db.execute("SELECT 1 FROM message_inbox WHERE message_id=? AND token=? AND status='RUNNING' AND julianday(lease_until)>julianday(?)",
                     (message_id, token, datetime.now(timezone.utc).isoformat())).fetchone()
    if row is None:
        raise StaleMessageClaim('Message queue lease expired or message already processed')
