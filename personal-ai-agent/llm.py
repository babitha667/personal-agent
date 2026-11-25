"""Native Ollama tool calling, with an opt-in legacy structured transport."""

import json
import os
import logging
from time import monotonic
from typing import Any, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from tool_protocol import output_schema, request_messages as structured_messages, decode_response


class LLMError(Exception):
    """Recoverable model connection or response failure."""


class RetryableLLMError(LLMError):
    """A transient transport failure that can be retried before side effects."""


class ModelTimeoutError(RetryableLLMError):
    pass


class ModelConnectionError(RetryableLLMError):
    pass


class EmptyModelResponse(LLMError):
    """No final text or tool call was produced; a bounded inference retry is safe."""


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, Any]], tools: list[dict],
             *, timeout: float | None = None, require_tool: bool = False) -> dict: ...


class OllamaClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self.model = os.getenv("OLLAMA_MODEL", "qwen3:0.6b")
        self.timeout = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))
        self.num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "1024"))
        self.think = os.getenv("OLLAMA_THINK", "1" if self.model == "qwen3:0.6b" else "0") == "1"
        self.num_ctx = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
        self.tool_mode = os.getenv('OLLAMA_TOOL_MODE', 'native')
        if self.tool_mode not in ('structured', 'native'):
            raise ValueError('OLLAMA_TOOL_MODE must be structured or native')
        if self.timeout <= 0 or self.num_predict <= 0 or self.num_ctx < 2048:
            raise ValueError("Invalid Ollama timeout or token limits")

    def chat(self, messages: list[dict[str, Any]], tools: list[dict],
             *, timeout: float | None = None, require_tool: bool = False) -> dict:
        request_timeout = min(self.timeout, timeout) if timeout is not None else self.timeout
        if request_timeout <= 0:
            raise ModelTimeoutError("Model time budget exhausted")
        # Qwen's legacy local templates can ignore think:false. The documented
        # soft switch is added to a copy, leaving saved messages unchanged.
        request_messages = [dict(message) for message in messages]
        if not self.think and self.model.lower().startswith('qwen3') and request_messages:
            system = next((item for item in request_messages if item.get('role') == 'system'), None)
            if system is not None and '/no_think' not in system.get('content', ''):
                system['content'] += '\n/no_think'
        payload_request = {
            "model": self.model, "messages": request_messages,
            "stream": False, "think": self.think, "keep_alive": "30m",
            "options": {"num_predict": self.num_predict, "num_ctx": self.num_ctx, "temperature": 0},
        }
        if self.tool_mode == 'structured':
            payload_request['messages'] = structured_messages(request_messages, tools)
            payload_request['format'] = output_schema(tools, require_tool=require_tool)
        else:
            payload_request['tools'] = tools
        request = Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload_request).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST",
        )
        started = monotonic()
        try:
            with urlopen(request, timeout=request_timeout) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise ValueError('Expected an Ollama response object')
            logging.getLogger(__name__).debug(
                "ollama_metrics model=%s wall_seconds=%.3f load_ns=%s prompt_tokens=%s "
                "prompt_eval_ns=%s generated_tokens=%s eval_ns=%s done_reason=%s",
                self.model, monotonic() - started, payload.get('load_duration'),
                payload.get('prompt_eval_count'), payload.get('prompt_eval_duration'),
                payload.get('eval_count'), payload.get('eval_duration'), payload.get('done_reason'),
            )
            if payload.get('done_reason') == 'length':
                raise LLMError("Model exceeded its output-token budget. No output was executed; inspect the log.")
            message = payload["message"]
            if not isinstance(message, dict) or message.get("role") != "assistant":
                raise ValueError("Invalid assistant message")
            content = message.get("content", "")
            calls = message.get("tool_calls", [])
            if not isinstance(content, str) or not isinstance(calls, list):
                raise ValueError("Invalid response fields")
            if not content.strip() and not calls:
                raise EmptyModelResponse("Ollama returned no final answer or tool call")
            if self.tool_mode == 'structured':
                return decode_response(message)
            return message
        except HTTPError as error:
            raise LLMError(
                f"Ollama returned HTTP {error.code}. Check `ollama list` and "
                f"that {self.model} is installed and supports tools."
            ) from error
        except TimeoutError as error:
            logging.getLogger(__name__).warning(
                "ollama_timeout model=%s elapsed_seconds=%.3f budget_seconds=%.3f prompt_characters=%s",
                self.model, monotonic() - started, request_timeout,
                sum(len(str(item.get('content', ''))) for item in payload_request['messages']),
            )
            raise ModelTimeoutError(f"Ollama exceeded the {request_timeout:.0f}s request budget. No plan was committed.") from error
        except (URLError, OSError) as error:
            raise ModelConnectionError("Could not reach Ollama. Check that the local server is running.") from error
        except (ValueError, KeyError, TypeError) as error:
            raise LLMError("Ollama returned an invalid or empty response.") from error


class DeepSeekClient:
    """DeepSeek Chat Completions with native tools and no credential logging."""

    tool_mode = 'native'

    def __init__(self) -> None:
        self.model = os.getenv('DEEPSEEK_MODEL', 'deepseek-flash')
        self.base_url = 'https://api.deepseek.com'
        self.timeout = float(os.getenv('DEEPSEEK_REQUEST_TIMEOUT_SECONDS', '120'))
        self.max_tokens = int(os.getenv('DEEPSEEK_MAX_TOKENS', '2048'))
        self.api_key = os.getenv('DEEPSEEK_API_KEY', '').strip()
        if not self.api_key:
            raise LLMError('Set DEEPSEEK_API_KEY in your terminal before starting the agent.')
        if self.timeout <= 0 or self.max_tokens <= 0:
            raise ValueError('DeepSeek timeout and token limit must be positive')

    def chat(self, messages: list[dict[str, Any]], tools: list[dict],
             *, timeout: float | None = None, require_tool: bool = False) -> dict:
        budget = min(self.timeout, timeout) if timeout is not None else self.timeout
        if budget <= 0:
            raise ModelTimeoutError('Model time budget exhausted')
        wire_messages = []
        for source in messages:
            item = {key: value for key, value in source.items()
                    if key in {'role', 'content', 'tool_calls', 'tool_call_id'}}
            if item.get('role') == 'assistant':
                if item.get('tool_calls'):
                    item['tool_calls'] = [dict(call, function=dict(call['function'], arguments=(
                        call['function']['arguments'] if isinstance(call['function']['arguments'], str)
                        else json.dumps(call['function']['arguments'])))) for call in item['tool_calls']]
            if item.get('role') == 'tool' and not item.get('tool_call_id'):
                raise LLMError('Tool results require their originating tool_call_id')
            wire_messages.append(item)
        body = {'model': self.model, 'messages': wire_messages, 'stream': False,
                'max_tokens': self.max_tokens, 'thinking': {'type': 'disabled'}}
        if tools:
            body['tools'] = tools
            body['tool_choice'] = 'auto'
        request = Request(self.base_url + '/chat/completions',
                          data=json.dumps(body).encode(),
                          headers={'Content-Type': 'application/json',
                                   'Authorization': 'Bearer ' + self.api_key}, method='POST')
        logger = logging.getLogger(__name__)
        started = monotonic()
        logger.info('deepseek_request model=%s endpoint=%s', self.model, request.full_url)
        try:
            with urlopen(request, timeout=budget) as response:
                payload = json.load(response)
            logger.info('deepseek_response model=%s response_id=%s wall_seconds=%.3f usage=%s',
                        self.model, payload.get('id'), monotonic() - started, payload.get('usage'))
            choice = payload['choices'][0]
            if choice.get('finish_reason') == 'length':
                raise LLMError('DeepSeek output token limit reached; no output was executed.')
            message = choice['message']
            if message.get('role') != 'assistant':
                raise ValueError('Expected assistant response')
            message['content'] = message.get('content') or ''
            if not isinstance(message['content'], str) or not isinstance(message.get('tool_calls', []), list):
                raise ValueError('Invalid assistant response fields')
            if not message['content'].strip() and not message.get('tool_calls'):
                raise EmptyModelResponse('DeepSeek returned no answer or tool calls')
            return {key: value for key, value in message.items()
                    if key in {'role', 'content', 'tool_calls'}}
        except HTTPError as error:
            if error.code == 429 or error.code >= 500:
                raise RetryableLLMError(f'DeepSeek temporarily unavailable (HTTP {error.code}).') from None
            raise LLMError(f'DeepSeek returned HTTP {error.code}; check API credentials, balance, and model access.') from None
        except TimeoutError:
            raise ModelTimeoutError(f'DeepSeek exceeded the {budget:.0f}s request budget.') from None
        except (URLError, OSError):
            raise ModelConnectionError('Could not reach the DeepSeek API.') from None
        except (ValueError, KeyError, IndexError, TypeError):
            raise LLMError('DeepSeek returned an invalid response.') from None


def create_client() -> ChatClient:
    provider = os.getenv('LLM_PROVIDER', 'deepseek').lower()
    if provider == 'deepseek':
        return DeepSeekClient()
    if provider == 'ollama':
        return OllamaClient()
    raise ValueError('LLM_PROVIDER must be deepseek or ollama')
