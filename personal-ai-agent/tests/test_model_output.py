import unittest
from model_output import parse_missing_information, parse_final_response


class ModelOutputTests(unittest.TestCase):
    def test_reply_with_thinking_wrapper(self):
        self.assertEqual(parse_final_response(
            'Greeting analysis</think>\n{"reply":"Hey!", "missing_information": []}'
        ), ([], "Hey!"))

    def test_invalid_reply_is_rejected(self):
        for reply in ('null', '123', '""'):
            with self.subTest(reply=reply), self.assertRaises(ValueError):
                parse_final_response('{"missing_information": [], "reply":' + reply + '}')

    def test_log_regression_reasoning_prefix_with_only_closing_tag(self):
        self.assertEqual(parse_missing_information(
            'The user sent a greeting. No functions needed.\n</think>\n\n{"missing_information": []}'
        ), [])

    def test_complete_thinking_wrapper_and_fenced_answer(self):
        self.assertEqual(parse_missing_information(
            '<think>Ignore this prefix {"not": "final"}</think>\n```json\n'
            '{"missing_information": ["Attendee"]}\n```'
        ), ['Attendee'])

    def test_unmarked_prose_incomplete_thinking_and_tool_text_are_rejected(self):
        for content in (
            'Hello!', '<think>{"missing_information": []}',
            'Analysis {"missing_information": []}',
            '{"missing_information": [], "tasks": []}',
            '{"tool_calls": [{"function": {"name": "apply_plan"}}]}',
            '{"missing_information": []} trailing text',
            '{"missing_information": "Who?"}',
        ):
            with self.subTest(content=content), self.assertRaises(ValueError):
                parse_missing_information(content)
