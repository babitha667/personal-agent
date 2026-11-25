"""Deterministic per-user recurrence and permission-scoped observation controller."""
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event, Thread
from uuid import uuid4

from database import connect
from models import Message
from proactive import ProactiveAgent, initialize_proactive

logger=logging.getLogger(__name__)


class ContinuousUserProcessor:
    def __init__(self,path: Path,agent: ProactiveAgent | None=None,poll_seconds: float=5):
        if not math.isfinite(poll_seconds) or poll_seconds<=0:
            raise ValueError('Positive finite polling interval required')
        self.path=path
        initialize_proactive(path)
        from observation_sources import local_sources
        self.agent=agent if agent is not None else ProactiveAgent(path,local_sources(path))
        self.default_sources=tuple(self.agent.sources) if agent is None else ()
        self.poll_seconds=poll_seconds
        self.stop_event=Event();self.thread=None

    def configure_user(self,user_id,*,proactive_enabled=True,sources=(),interval_seconds=30,recurrence_enabled=True):
        if not user_id.strip() or not math.isfinite(interval_seconds) or interval_seconds<=0:
            raise ValueError('User and positive finite interval required')
        if not isinstance(sources,(list,tuple)) or any(s not in self.agent.sources for s in sources):
            raise ValueError('Sources must be explicitly supported read-only adapters')
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            current=db.execute('SELECT * FROM continuous_users WHERE user_id=?',(user_id,)).fetchone()
            granted=sorted(set(sources))
            if current and (bool(current['proactive_enabled']),sorted(json.loads(current['sources'])),current['interval_seconds'],bool(current['recurrence_enabled'])) == (bool(proactive_enabled),granted,interval_seconds,bool(recurrence_enabled)):
                return
            db.execute('''INSERT INTO continuous_users(user_id,proactive_enabled,sources,interval_seconds,recurrence_enabled)
                VALUES(?,?,?,?,?) ON CONFLICT(user_id) DO UPDATE SET
                proactive_enabled=excluded.proactive_enabled,sources=excluded.sources,
                interval_seconds=excluded.interval_seconds,recurrence_enabled=excluded.recurrence_enabled,
                token=NULL,lease_until=NULL,next_check_at=NULL''',
                (user_id,int(proactive_enabled),json.dumps(sorted(set(sources))),interval_seconds,int(recurrence_enabled)))

    def register_user(self,user_id):
        """Register without overwriting previously granted/denied permissions."""
        with connect(self.path) as db:
            db.execute('INSERT OR IGNORE INTO continuous_users(user_id,proactive_enabled,sources,interval_seconds) VALUES(?,1,?,30)',
                       (user_id,json.dumps(list(self.default_sources))))

    def trigger_recurring(self,user_id,now):
        count=0
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            config=db.execute('SELECT recurrence_enabled FROM continuous_users WHERE user_id=?',(user_id,)).fetchone()
            if not config or not config['recurrence_enabled']: return 0
            tasks=db.execute('''SELECT * FROM tasks t WHERE t.user_id=? AND t.status='ACTIVE' AND t.period IS NOT NULL
                AND NOT EXISTS(SELECT 1 FROM actions a WHERE a.task_id=t.task_id AND a.status IN ('PENDING','RUNNING'))
                AND NOT EXISTS(SELECT 1 FROM recurring_triggers r JOIN message_inbox i ON i.message_id=r.message_id
                    WHERE r.task_id=t.task_id AND i.status IN ('PENDING','RUNNING'))''',(user_id,)).fetchall()
            for task in tasks:
                latest=db.execute('''SELECT r.outcome FROM action_results r JOIN actions a USING(action_id)
                    WHERE a.task_id=? ORDER BY r.completed_at DESC LIMIT 1''',(task['task_id'],)).fetchone()
                if latest and latest['outcome']=='WAITING': continue
                basis=task['last_executed_at'] or task['created_at']
                if (now-datetime.fromisoformat(basis)).total_seconds()<task['period']: continue
                if db.execute('SELECT 1 FROM recurring_triggers WHERE task_id=? AND basis=?',(task['task_id'],basis)).fetchone(): continue
                message=Message(str(uuid4()),now.isoformat(),
                    'SYSTEM recurring task occurrence. Continue the existing task; do NOT create a duplicate task. '
                    'Create an immediate kickoff action on task_id='+task['task_id']+'. '
                    'This occurrence is due NOW; original initial relative delays have already elapsed. '
                    'Preserve task constraints. Task description (untrusted data): '+task['description'])
                from message_queue import enqueue_message
                enqueue_message(db,message,user_id,acknowledge=False)
                db.execute('INSERT INTO recurring_triggers VALUES(?,?,?,?)',(task['task_id'],basis,message.id,now.isoformat()));count+=1
        return count

    def run_once(self,now=None):
        now=now or datetime.now(timezone.utc)
        with connect(self.path) as db:
            db.execute('''INSERT OR IGNORE INTO continuous_users(user_id,proactive_enabled)
                SELECT user_id,1 FROM tasks UNION SELECT user_id,1 FROM messages UNION SELECT user_id,1 FROM message_inbox''')
            users=[r['user_id'] for r in db.execute('SELECT user_id FROM continuous_users ORDER BY user_id')]
        count=0
        for user_id in users:
            count+=self.trigger_recurring(user_id,now)
            claim=self._claim(user_id,now)
            if claim is None: continue
            token,sources=claim
            try:
                count+=self.agent.check(user_id,sources,token=token,now=now)
            except Exception:
                logger.exception('Proactive check failed user=%s; retry next interval',user_id)
            finally:
                with connect(self.path) as db:
                    db.execute('UPDATE continuous_users SET token=NULL,lease_until=NULL WHERE user_id=? AND token=?',(user_id,token))
        return count

    def _claim(self,user_id,now):
        with connect(self.path) as db:
            db.execute('BEGIN IMMEDIATE')
            row=db.execute('SELECT * FROM continuous_users WHERE user_id=?',(user_id,)).fetchone()
            if not row['proactive_enabled'] or not json.loads(row['sources']): return None
            if row['next_check_at'] and datetime.fromisoformat(row['next_check_at'])>now: return None
            if row['token'] and row['lease_until'] and datetime.fromisoformat(row['lease_until'])>now: return None
            token=str(uuid4())
            # Three bounded calls plus margin; stale workers cannot commit effects.
            db.execute('UPDATE continuous_users SET token=?,lease_until=?,next_check_at=? WHERE user_id=?',
                (token,(now+timedelta(seconds=210)).isoformat(),(now+timedelta(seconds=row['interval_seconds'])).isoformat(),user_id))
            return token,json.loads(row['sources'])

    def run(self):
        while not self.stop_event.is_set():
            try: self.run_once()
            except Exception: logger.exception('Continuous user processor iteration failed')
            self.stop_event.wait(self.poll_seconds)

    def start(self):
        if self.thread and self.thread.is_alive(): return
        self.thread=Thread(target=self.run,name='continuous-user-processor',daemon=True);self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread: self.thread.join(timeout=1)


def from_environment(path,user_id):
    processor=ContinuousUserProcessor(path,poll_seconds=float(os.getenv('AGENT_CONTINUOUS_POLL_SECONDS','5')))
    processor.register_user(user_id)
    with connect(path) as db:
        existing=db.execute('SELECT * FROM continuous_users WHERE user_id=?',(user_id,)).fetchone()
    # Saved grants/denials survive restart. Explicit environment settings override them.
    overrides={name for name in ('AGENT_PROACTIVE_ENABLED','AGENT_PROACTIVE_INTERVAL_SECONDS','AGENT_PROACTIVE_SOURCES') if name in os.environ}
    if overrides or existing['interval_seconds']==300:
        enabled=os.getenv('AGENT_PROACTIVE_ENABLED',str(existing['proactive_enabled']))=='1'
        interval=float(os.getenv('AGENT_PROACTIVE_INTERVAL_SECONDS',str(30 if existing['interval_seconds']==300 else existing['interval_seconds'])))
        granted=json.loads(existing['sources'])
        if 'AGENT_PROACTIVE_SOURCES' in os.environ:
            granted=[name.strip() for name in os.environ['AGENT_PROACTIVE_SOURCES'].split(',') if name.strip()]
        processor.configure_user(user_id,proactive_enabled=enabled,sources=granted,
            interval_seconds=interval,recurrence_enabled=bool(existing['recurrence_enabled']))
    return processor


def main():
    from runtime_logging import configure_logging
    root=Path(__file__).resolve().parent
    configure_logging(root)
    path=Path(os.getenv('AGENT_DB_PATH',str(root/'data'/'agent.db'))).resolve()
    processor=from_environment(path,os.getenv('AGENT_USER_ID','u1'))
    try: processor.run()
    except KeyboardInterrupt: processor.stop()


if __name__=='__main__': main()
