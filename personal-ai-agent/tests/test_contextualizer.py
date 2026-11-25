import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from urllib.error import URLError

from legacy_contextualizer import Contextualizer
from agent_tools import AgentTools, TOOL_DEFINITIONS, PLANNER_TOOLS
from database import connect
from llm import LLMError, OllamaClient
from models import Message
from storage import MessageStore


# Explicit legacy tool registry for transaction/backward-compatibility tests.
TEST_TOOLS = PLANNER_TOOLS + TOOL_DEFINITIONS

def call(name, arguments):
    return {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": name, "arguments": arguments}}
    ]}


def done(missing=None):
    if missing:
        return call('ask_clarification', {'reply': missing[0], 'missing_information': missing})
    return call('respond_chat', {'reply': 'Hello'})


def plan():
    return {"cancel_task_ids": [], "tasks": [{
        "description": "Drink water", "period": 86400,
        "actions": [{"exec_at": None, "description": "Send water reminder"}],
    }]}


class ContextualizerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "agent.db"
        self.store = MessageStore(self.path)
        self.llm = Mock()
        self.contextualizer = Contextualizer(self.store, llm=self.llm, tool_definitions=TEST_TOOLS)

    def test_message_is_passed_with_tools_without_preloaded_tasks(self):
        self.llm.chat.return_value = done()
        context = self.contextualizer.receive_message(Message.from_content("there?"))
        messages = self.llm.chat.call_args.args[0]
        self.assertEqual(messages[1]["role"], "user")
        self.assertTrue(messages[1]["content"].endswith("there?"))
        self.assertEqual(context.tasks, [])
        self.assertIn("create_reminder", [tool["function"]["name"] for tool in self.llm.chat.call_args.kwargs["tools"]])
        self.assertNotIn('"task_id"', messages[0]["content"])

    def test_acknowledgment_precedes_storage_and_inference(self):
        events = []
        original_save = self.store.save

        def save(*args, **kwargs):
            events.append("save")
            return original_save(*args, **kwargs)

        def infer(*args, **kwargs):
            events.append("inference")
            return done()

        self.llm.chat.side_effect = infer
        contextualizer = Contextualizer(
            self.store, llm=self.llm, tool_definitions=TEST_TOOLS,
            on_acknowledge=lambda text: events.append(text),
        )
        with patch.object(self.store, "save", side_effect=save):
            contextualizer.receive_message(Message.from_content("there?"))
        self.assertEqual(events, ["👍 ACK", "save", "inference", "save"])

    def test_disabled_search_is_rejected_before_valid_plan(self):
        self.llm.chat.side_effect = [call("find_tasks", {"query": ""}), call("apply_plan", plan())]
        result = self.contextualizer.receive_message(Message.from_content("Remind me to drink water"))
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(result.actions[0].task_id, result.tasks[0].task_id)
        self.assertEqual(result.actions[0].status, "PENDING")
        messages = self.llm.chat.call_args.args[0]
        read_result = next(item for item in messages if item.get("tool_name") == "find_tasks")
        self.assertEqual(json.loads(read_result["content"])["error"], "Unknown tool: find_tasks")
        self.assertEqual(self.llm.chat.call_count, 2)
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0], 1)

    def test_unknown_tool_error_is_returned_to_model(self):
        self.llm.chat.side_effect = [call("run_sql", {"sql": "DROP TABLE tasks"}), done(["Which reminder?"])]
        result = self.contextualizer.receive_message(Message.from_content("Change it"))
        self.assertEqual(result.missing_information, ["Which reminder?"])
        messages = self.llm.chat.call_args.args[0]
        self.assertIn("Unknown tool", messages[3]["content"])

    def test_existing_tasks_are_preloaded_and_new_work_needs_one_call(self):
        from planning import save_plan, validate_plan
        tasks, actions = validate_plan(plan()['tasks'], 'u1')
        save_plan(self.path, tasks, actions)
        self.llm.chat.return_value = call('apply_plan', plan())
        result = self.contextualizer.receive_message(Message.from_content('Nudge me to stretch'))
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(self.llm.chat.call_count, 1)
        prompt = self.llm.chat.call_args.args[0][0]['content']
        self.assertIn(tasks[0].task_id, prompt)
        self.assertIn(actions[0].exec_at, prompt)

    def test_empty_system_can_create_task_in_one_call(self):
        self.llm.chat.return_value = call('apply_plan', plan())
        result = self.contextualizer.receive_message(Message.from_content('Remind me to drink water in two minutes'))
        self.assertEqual(len(result.tasks), 1)
        self.assertEqual(self.llm.chat.call_count, 1)
        names = [tool['function']['name'] for tool in self.llm.chat.call_args.kwargs['tools']]
        self.assertIn('create_reminder', names)

    def test_invalid_plan_can_be_corrected_without_partial_write(self):
        invalid = plan()
        invalid["tasks"][0]["period"] = -1
        self.llm.chat.side_effect = [call("find_tasks", {"query": ""}), call("apply_plan", invalid), call("apply_plan", plan())]
        result = self.contextualizer.receive_message(Message.from_content("Water reminder"))
        self.assertEqual(len(result.tasks), 1)
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 1)

    def test_missing_information_and_model_failure_preserve_input(self):
        self.llm.chat.return_value = done(["Meeting attendee"])
        result = self.contextualizer.receive_message(Message.from_content("Schedule a meeting"))
        self.assertEqual(result.missing_information, ["Meeting attendee"])
        self.llm.chat.side_effect = LLMError("offline")
        with self.assertRaises(LLMError):
            self.contextualizer.receive_message(Message.from_content("hello"))
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM messages WHERE role='user'").fetchone()[0], 2)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)

    def test_tool_loop_is_bounded(self):
        self.llm.chat.return_value = call("find_tasks", {"query": ""})
        with self.assertRaisesRegex(LLMError, "step limit"):
            self.contextualizer.receive_message(Message.from_content("Loop"))
        self.assertEqual(self.llm.chat.call_count, 3)

    def test_native_greeting_finishes_without_forcing_a_tool(self):
        self.llm.chat.return_value = {'role': 'assistant', 'content': 'Hello! How can I help?'}
        result = self.contextualizer.receive_message(Message.from_content('hey there'))
        self.assertEqual(result.reply, 'Hello! How can I help?')
        self.assertEqual(self.llm.chat.call_count, 1)
        self.assertEqual(result.tasks, [])

    def test_casual_message_gets_chatbot_reply_without_writes(self):
        self.llm.chat.return_value = call('respond_chat', {'reply': "Hey! What's on your mind?"})
        result = self.contextualizer.receive_message(Message.from_content("what's up?"))
        self.assertEqual(result.reply, "Hey! What's on your mind?")
        self.assertEqual(result.tasks, [])
        self.assertEqual(self.llm.chat.call_count, 1)
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM actions").fetchone()[0], 0)

    def test_unclear_task_can_ask_a_question_without_creating_work(self):
        self.llm.chat.return_value = call('ask_clarification', {'reply': 'Who should attend the meeting?', 'missing_information': ['Meeting attendee']})
        result = self.contextualizer.receive_message(Message.from_content("Schedule a meeting"))
        self.assertEqual(result.reply, "Who should attend the meeting?")
        self.assertEqual(result.missing_information, ["Meeting attendee"])
        self.assertEqual(result.tasks, [])

    def test_native_text_does_not_execute_described_tools(self):
        self.llm.chat.return_value = {'role': 'assistant', 'content': 'Here is how to create a reminder.'}
        result = self.contextualizer.receive_message(Message.from_content('Explain reminders'))
        self.assertEqual(result.tasks, [])
        self.assertEqual(self.llm.chat.call_count, 1)

    def test_native_general_answer_returns_in_one_call(self):
        self.llm.chat.return_value = {'role': 'assistant', 'content': 'The capital of France is Paris.'}
        result = self.contextualizer.receive_message(Message.from_content('Capital of France?'))
        self.assertIn('Paris', result.reply)
        self.assertEqual(self.llm.chat.call_count, 1)

    def test_mixed_write_batches_are_rejected(self):
        response = call("find_tasks", {"query": ""})
        response["tool_calls"] += call("apply_plan", plan())["tool_calls"]
        self.llm.chat.side_effect = [response, done(["Retry planning"])]
        result = self.contextualizer.receive_message(Message.from_content("Water"))
        self.assertEqual(result.tasks, [])
        with connect(self.path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0], 0)

    def test_argument_validation_blocks_user_override(self):
        tools = AgentTools(self.path, "u1", Message.from_content("hello"))
        for name, arguments in [("find_tasks", {"query": "", "user_id": "u2"}),
                                ("get_task", {"task_id": 123}), ("find_tasks", []),
                                ("apply_plan", {"tasks": []})]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                tools.execute(name, arguments)

    def test_ollama_serializes_native_tools_and_accepts_contentless_calls(self):
        message = call("find_tasks", {"query": ""})
        del message["content"]
        response = io.BytesIO(json.dumps({"message": message}).encode())
        with patch.dict("os.environ", {"OLLAMA_BASE_URL": "http://localhost:11434", "OLLAMA_MODEL": "qwen3:4b", "OLLAMA_TOOL_MODE": "native"}):
            client = OllamaClient()
        with patch("llm.urlopen", return_value=response) as transport:
            self.assertEqual(client.chat([], TOOL_DEFINITIONS), message)
        request = transport.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(request.full_url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["tools"], TOOL_DEFINITIONS)
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertNotIn("format", payload)

    def test_qwen_small_model_uses_native_tools_and_reasoning_without_json_format(self):
        from agent_tools import PLANNER_TOOLS
        response=io.BytesIO(json.dumps({'message':{'role':'assistant','content':'Hello!'}}).encode())
        with patch.dict('os.environ',{'OLLAMA_MODEL':'qwen3:0.6b','OLLAMA_TOOL_MODE':'native','OLLAMA_THINK':'1','OLLAMA_NUM_PREDICT':'1024'}):
            client=OllamaClient()
        with patch('llm.urlopen',return_value=response) as transport:
            result=client.chat([{'role':'user','content':'hey there'}],PLANNER_TOOLS)
        payload=json.loads(transport.call_args.args[0].data)
        self.assertTrue(payload['think'])
        self.assertEqual(payload['tools'],PLANNER_TOOLS)
        self.assertNotIn('format',payload)
        self.assertEqual(result['content'],'Hello!')

    def test_ollama_connection_and_invalid_response_errors(self):
        with patch("llm.urlopen", side_effect=URLError("offline")):
            with self.assertRaises(LLMError):
                OllamaClient().chat([], TOOL_DEFINITIONS)

    def test_qwen_speed_controls_and_prompt_copy(self):
        original = [{"role": "system", "content": "Use tools."}, {"role": "user", "content": "Water reminder"}]
        response = io.BytesIO(json.dumps({"message": done()}).encode())
        with patch.dict('os.environ', {"OLLAMA_MODEL": "qwen3:4b", "OLLAMA_NUM_PREDICT": "512",
                                      "OLLAMA_NUM_CTX": "8192", "OLLAMA_REQUEST_TIMEOUT_SECONDS": "30", "OLLAMA_TOOL_MODE": "native"}):
            client = OllamaClient()
        with patch('llm.urlopen', return_value=response) as transport:
            client.chat(original, TOOL_DEFINITIONS, timeout=12)
        payload = json.loads(transport.call_args.args[0].data)
        self.assertIn('/no_think', payload['messages'][0]['content'])
        self.assertEqual(original[0]['content'], 'Use tools.')
        self.assertFalse(payload['think'])
        self.assertEqual(payload['options']['num_predict'], 512)
        self.assertEqual(payload['options']['num_ctx'], 8192)
        self.assertEqual(payload['keep_alive'], '30m')
        self.assertEqual(transport.call_args.kwargs['timeout'], 12)

    def test_planning_deadline_prevents_late_write(self):
        clock = [0.0]

        def late_inference(*args, **kwargs):
            clock[0] = 31.0
            return call('apply_plan', plan())

        self.llm.chat.side_effect = late_inference
        self.contextualizer.planning_timeout = 30
        with patch('legacy_contextualizer.monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(LLMError, 'time budget'):
                self.contextualizer.receive_message(Message.from_content('Water reminder'))
        with connect(self.path) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM tasks').fetchone()[0], 0)
        with patch("llm.urlopen", return_value=io.BytesIO(b"{}")):
            with self.assertRaises(LLMError):
                OllamaClient().chat([], TOOL_DEFINITIONS)
