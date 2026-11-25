"""Flat, database-backed history: every past task action appears once."""
from action_data import parse_context


def snapshot(action, result=None):
    action = dict(action)
    facts = dict(parse_context(action.get('context', {})))
    facts.pop('past_actions', None)
    entry = {key:action.get(key) for key in (
        'action_id','task_id','description','exec_at','status','created_at','attempt_count')}
    entry['context'] = facts
    entry.update(outcome=action.get('status'),reason=None,completed_at=None,tool_results={})
    if result is not None:
        entry.update({key:result.get(key) for key in ('outcome','reason','completed_at')})
        entry['tool_results'] = parse_context(result.get('details', {}))
    return entry


def task_history(db, task_id):
    if task_id is None:
        return []
    history = []
    for row in db.execute("SELECT * FROM actions WHERE task_id=? AND status IN ('COMPLETED','FAILED','CANCELLED') ORDER BY created_at,action_id", (task_id,)):
        result = None
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='action_results'").fetchone():
            saved = db.execute('SELECT * FROM action_results WHERE action_id=?',(row['action_id'],)).fetchone()
            result = dict(saved) if saved else None
        history.append(snapshot(row,result))
    return history


def with_history(db, task_id, context):
    facts = dict(parse_context(context))
    # Never accept model-invented history or truncate a real task's history.
    facts['past_actions'] = task_history(db,task_id)
    return facts
