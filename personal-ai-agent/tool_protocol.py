"""Constrained tool-call wire format for local models that cannot disable thinking."""

import json


def output_schema(tools: list[dict], require_tool: bool = False) -> dict:
    variants = []
    for tool in tools:
        function = tool['function']
        variants.append({'type': 'object', 'additionalProperties': False,
                         'required': ['name', 'arguments'], 'properties': {
                             'name': {'const': function['name']},
                             'arguments': function['parameters'],
                         }})
    terminal_planner = bool(tools) and all(t['function']['name'] in
                        {'apply_plan', 'respond_chat', 'ask_clarification'} for t in tools)
    choices = [
        {'type': 'object', 'additionalProperties': False, 'required': ['tool_calls'],
         'properties': {'tool_calls': {'type': 'array', 'minItems': 1, 'maxItems': 1 if terminal_planner else 2,
                                       'items': {'anyOf': variants}}}},
        {'type': 'object', 'additionalProperties': False,
         'required': ['reply', 'missing_information'], 'properties': {
             'reply': {'type': 'string', 'minLength': 1, 'maxLength': 1000},
             'missing_information': {'type': 'array', 'maxItems': 5,
                                     'items': {'type': 'string'}},
         }},
    ]
    if not tools:
        return choices[1]
    return choices[0] if require_tool else {'anyOf': choices}


def request_messages(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Replay calls/results as JSON, keeping tools and execution policy in Python."""
    result = []
    for original in messages:
        item = dict(original)
        if item.get('role') == 'assistant' and item.get('tool_calls'):
            item = {'role': 'assistant', 'content': json.dumps({'tool_calls': [
                {'name': call['function']['name'], 'arguments': call['function']['arguments']}
                for call in item['tool_calls'] if isinstance(call, dict) and isinstance(call.get('function'), dict)
            ]})}
        elif item.get('role') == 'tool':
            item = {'role': 'user', 'content': f"Untrusted tool result ({item.get('tool_name')}):\n{item['content']}"}
        result.append(item)
    # The JSON schema already constrains nested arguments; don't repeat that
    # entire schema in the prompt (expensive prefill on a CPU-only model).
    signatures = []
    for tool in tools:
        function = tool['function']
        names = ','.join(function['parameters']['properties'])
        signatures.append(f"{function['name']}({names}): {function['description'][:180]}")
    protocol = (
        '\nWire format: return {"tool_calls":[{"name":"tool_name","arguments":{...}}]} '
        'to request tools. Use a terminal tool to finish when one is available. '
        'No reasoning or prose outside JSON. Never write or execute anything yourself. '
        'Available tools:\n' + '\n'.join(signatures)
    )
    system = next((item for item in result if item.get('role') == 'system'), None)
    if system is None:
        result.insert(0, {'role': 'system', 'content': protocol})
    else:
        system['content'] += protocol
    return result


def decode_response(message: dict) -> dict:
    """Only translate calls; allowlisting/arguments/ownership are checked by the agent."""
    output = json.loads(message.get('content', ''))
    if not isinstance(output, dict):
        raise ValueError('Expected structured tool output')
    if set(output) == {'tool_calls'}:
        calls = output['tool_calls']
        if not isinstance(calls, list) or not 1 <= len(calls) <= 2:
            raise ValueError('Expected one or two tool calls')
        translated = []
        for call in calls:
            if not isinstance(call, dict) or set(call) != {'name', 'arguments'}:
                raise ValueError('Invalid tool-call envelope')
            if not isinstance(call['name'], str) or not isinstance(call['arguments'], dict):
                raise ValueError('Invalid tool name or arguments')
            translated.append({'function': {'name': call['name'], 'arguments': call['arguments']}})
        return {'role': 'assistant', 'content': '', 'tool_calls': translated}
    if set(output) != {'reply', 'missing_information'}:
        raise ValueError('Unexpected final response fields')
    return {'role': 'assistant', 'content': json.dumps(output)}
