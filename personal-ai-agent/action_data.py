"""Action context serialization shared by stores and queues."""
import json
from dataclasses import asdict

from models import Action


def parse_context(value):
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError('Action context must be an object of relevant facts and constraints')
    json.dumps(value, allow_nan=False)
    return value


def encode_context(value):
    return json.dumps(parse_context(value), ensure_ascii=False, allow_nan=False)


def action_record(action):
    record = asdict(action)
    record['context'] = encode_context(action.context)
    return record


def action_from_record(row):
    data = {key: row[key] for key in Action.__dataclass_fields__}
    data['context'] = parse_context(data['context'])
    return Action(**data)


def action_from_saved(record):
    record = dict(record)
    if 'instructions' in record:
        previous = record.pop('instructions')
        record.setdefault('description',previous)
    context = parse_context(record.get('context', {}))
    for old, key in (('notification','legacy_message'), ('scheduled_for','requested_time')):
        value = record.pop(old, None)
        if value is not None:
            context.setdefault(key, value)
    record['context'] = context
    return Action(**record)


def merge_context(base, updates):
    """Preserve existing nested facts unless a new value explicitly replaces one."""
    merged = json.loads(encode_context(base))
    for key,value in parse_context(updates).items():
        if key == 'past_actions':
            continue  # History is derived from committed database records.
        if isinstance(value,dict) and isinstance(merged.get(key),dict):
            merged[key] = merge_context(merged[key],value)
        else:
            merged[key] = value
    return merged


def context_input(value):
    if not isinstance(value,dict):
        raise ValueError('Supply context as a JSON object, not a JSON-encoded string')
    return parse_context(value)
