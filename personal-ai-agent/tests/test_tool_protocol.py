import io
import json
import os
import unittest
from unittest.mock import patch

from agent_tools import TOOL_DEFINITIONS
from llm import OllamaClient, LLMError
from tool_protocol import decode_response, request_messages, output_schema


class ToolProtocolTests(unittest.TestCase):
    def test_required_tool_mode_cannot_decode_a_chat_promise(self):
        schema = output_schema(TOOL_DEFINITIONS, require_tool=True)
        self.assertEqual(schema['required'], ['tool_calls'])
        self.assertNotIn('reply', schema['properties'])

    def test_structured_calls_become_agent_tool_calls(self):
        response = {'role': 'assistant', 'content': json.dumps({'tool_calls': [
            {'name': 'find_tasks', 'arguments': {'query': 'water'}}
        ]})}
        decoded = decode_response(response)
        self.assertEqual(decoded['tool_calls'][0]['function'],
                         {'name': 'find_tasks', 'arguments': {'query': 'water'}})

    def test_wire_history_conversion_does_not_change_agent_history(self):
        messages = [
            {'role': 'system', 'content': 'Use tools'},
            {'role': 'assistant', 'tool_calls': [{'function': {'name': 'find_tasks', 'arguments': {'query': 'water'}}}]},
            {'role': 'tool', 'tool_name': 'find_tasks', 'content': '{"tasks": []}'},
        ]
        encoded = request_messages(messages, TOOL_DEFINITIONS)
        self.assertEqual(messages[0]['content'], 'Use tools')
        self.assertIn('tool_calls', json.loads(encoded[1]['content']))
        self.assertIn('Untrusted tool result', encoded[2]['content'])
        self.assertNotIn('find_tasks(query)', encoded[0]['content'])
        self.assertIn('respond_chat(reply)', encoded[0]['content'])

    def test_default_transport_constrains_tool_calls(self):
        output = {'tool_calls': [{'name': 'find_tasks', 'arguments': {'query': 'water'}}]}
        response = io.BytesIO(json.dumps({'message': {'role': 'assistant', 'content': json.dumps(output)}}).encode())
        with patch.dict(os.environ, {'OLLAMA_TOOL_MODE': 'structured'}):
            client = OllamaClient()
        with patch('llm.urlopen', return_value=response) as request:
            result = client.chat([{'role': 'system', 'content': 'Plan'}], TOOL_DEFINITIONS)
        sent = json.loads(request.call_args.args[0].data)
        self.assertIn('anyOf', sent['format'])
        self.assertNotIn('tools', sent)
        self.assertEqual(result['tool_calls'][0]['function']['name'], 'find_tasks')

    def test_malformed_and_truncated_outputs_are_never_executed(self):
        for content in ('{"tool_calls": "bad"}', '{"tasks": []}', '{"tool_calls": [{"name":"run_sql"}]}'):
            with self.subTest(content=content), self.assertRaises(ValueError):
                decode_response({'content': content})
        payload = {'done_reason': 'length', 'message': {'role': 'assistant', 'content': '{'}}
        with patch('llm.urlopen', return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(LLMError, 'output-token'):
                OllamaClient().chat([], TOOL_DEFINITIONS)
