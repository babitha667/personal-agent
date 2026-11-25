import io
import json
import unittest
from unittest.mock import patch
from llm import DeepSeekClient, create_client, LLMError


class DeepSeekTests(unittest.TestCase):
    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-secret'}, clear=True)
    def test_native_tools_and_correlated_results(self):
        client = create_client()
        self.assertIsInstance(client, DeepSeekClient)
        calls = [{'id':'c1','type':'function','function':{'name':'cancel_task','arguments':{'confirm':True}}}]
        history = [{'role':'assistant','content':'','tool_calls':calls},
                   {'role':'tool','tool_name':'cancel_task','tool_call_id':'c1','content':'{}'}]
        response = {'choices':[{'finish_reason':'stop','message':{'role':'assistant','content':'Done.'}}]}
        with patch('llm.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            self.assertEqual(client.chat(history, [])['content'], 'Done.')
        request = send.call_args.args[0]
        body = json.loads(request.data)
        self.assertEqual(body['model'], 'deepseek-flash')
        self.assertEqual(body['thinking'], {'type':'disabled'})
        self.assertNotIn('tools', body)
        self.assertNotIn('tool_name', body['messages'][1])
        self.assertEqual(body['messages'][1]['tool_call_id'], 'c1')
        self.assertEqual(json.loads(body['messages'][0]['tool_calls'][0]['function']['arguments']), {'confirm':True})
        self.assertIsInstance(calls[0]['function']['arguments'], dict)

    @patch.dict('os.environ', {}, clear=True)
    def test_missing_key_fails_before_network(self):
        with self.assertRaisesRegex(LLMError, 'DEEPSEEK_API_KEY'):
            create_client()

    @patch.dict('os.environ', {'DEEPSEEK_API_KEY': 'test-secret'}, clear=True)
    def test_auto_tool_selection_allows_chat(self):
        tools = [{'type':'function','function':{'name':'example','parameters':{'type':'object'}}}]
        response = {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[
            {'id':'c1','type':'function','function':{'name':'example','arguments':'{}'}}]}}]}
        with patch('llm.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as send:
            result = DeepSeekClient().chat([{'role':'user','content':'hello'}], tools, require_tool=True)
        body = json.loads(send.call_args.args[0].data)
        self.assertEqual(body['tools'], tools)
        self.assertEqual(body['tool_choice'], 'auto')
        self.assertEqual(result['tool_calls'][0]['id'], 'c1')
