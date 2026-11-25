"""Tool-using agent for a single claimed action; never waits for future work."""

import json
import logging
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from action_queue import ActionQueue, Claim, StaleClaim
from action_data import action_from_record
from database import connect
from executor_tools import EXECUTOR_TOOLS, ExecutorTools
from executor_management import MANAGEMENT_NAMES, ManagementConflict
from llm import ChatClient, LLMError, create_client

logger = logging.getLogger(__name__)


class ActionExecutor:
    def __init__(self, queue: ActionQueue, llm: ChatClient | None = None,
                 timezone: str = 'Asia/Kolkata') -> None:
        self.queue = queue
        self.llm = llm if llm is not None else create_client()
        self.timezone = ZoneInfo(timezone)

    def execute(self, claim: Claim) -> None:
        task = self.queue.context(claim)
        # Use the latest owned, fenced state, including context enrichment.
        with connect(self.queue.path) as db:
            row = db.execute('SELECT * FROM actions WHERE action_id=?',(claim.action.action_id,)).fetchone()
            claim = Claim(action_from_record(row), claim.token)
        tools = ExecutorTools(self.queue, claim)
        prompt = (Path(__file__).parent / 'prompts' / 'action_executor.txt').read_text()
        messages = [{'role': 'system', 'content': prompt +
                     f'\nCurrent time: {datetime.now(self.timezone).isoformat()}. Timezone: {self.timezone.key}.'},
                    {'role': 'user', 'content': json.dumps({'claimed_action': asdict(claim.action), 'task': task})}]
        total = 0
        for step in range(7):
            logger.debug('executor_llm_start action=%s step=%s', claim.action.action_id, step + 1)
            allowed = {'read_actions', 'add_actions', 'send_message', 'finish_action', 'send_email', 'read_email', 'draft_email'} | MANAGEMENT_NAMES
            if tools.notification is not None:
                allowed.discard('send_message')
            if tools.email is not None or tools.management.task_id is None:
                allowed.discard('send_email')
                allowed.discard('draft_email')
            if tools.follow_ups or tools.management.task_id is None:
                allowed.discard('add_actions')
            if tools.management.task_id is not None:
                allowed.discard('create_task')
                allowed.discard('select_task')
            if step == 6:
                allowed = {'finish_action'}
                messages.append({'role':'system','content':'Execution budget reached. Call finish_action now to commit verified staged work, or WAITING if blocked. Do not claim unstaged success.'})
            available = [tool for tool in EXECUTOR_TOOLS if tool['function']['name'] in allowed]
            response = self.llm.chat(messages, tools=available)
            logger.debug('executor_llm_result action=%s step=%s response=%s', claim.action.action_id, step + 1, json.dumps(response))
            if not isinstance(response, dict) or response.get('role') != 'assistant':
                raise LLMError('Invalid executor model response')
            calls = response.get('tool_calls', [])
            if not isinstance(calls, list):
                raise LLMError('Invalid executor tool calls')
            messages.append(response)
            if not calls:
                messages.append({'role': 'system', 'content': 'Use native tools. Stage the necessary effect, then finish_action. Do not respond in prose.'})
                continue
            snapshot = (tools.notification, list(tools.follow_ups), tools.email, set(tools.read_email_ids), tools.email_read_performed, deepcopy(tools.management))
            batch_failed = False
            for call_index, call in enumerate(calls):
                total += 1
                if total > 16:
                    raise LLMError('Executor tool-call limit reached')
                name = 'invalid_tool'
                try:
                    if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                        raise ValueError('Invalid tool call')
                    function = call['function']
                    name = function.get('name')
                    if not isinstance(name, str):
                        raise ValueError('Tool name must be text')
                    args = function.get('arguments')
                    if isinstance(args, str):
                        args = json.loads(args)
                    if name not in allowed:
                        raise ValueError('Tool is not available in the current execution state')
                    if name == 'finish_action' and (call_index != len(calls) - 1 or batch_failed):
                        raise ValueError('finish_action must run last in a valid batch')
                    logger.info('Executor tool: %s [action %s]', name, claim.action.action_id)
                    logger.debug('executor_tool_requested action=%s name=%s arguments=%s', claim.action.action_id, name, json.dumps(args))
                    result = tools.execute(name, args)
                except StaleClaim:
                    raise
                except ManagementConflict as error:
                    raise LLMError(str(error)) from error
                except (ValueError, TypeError) as error:
                    batch_failed = True
                    result = {'error': str(error)}
                logger.debug('executor_tool_result action=%s name=%s result=%s', claim.action.action_id, name, json.dumps(result))
                tool_result = {'role': 'tool', 'tool_name': name, 'content': json.dumps(result)}
                if isinstance(call, dict) and isinstance(call.get('id'), str):
                    tool_result['tool_call_id'] = call['id']
                messages.append(tool_result)
                if tools.finished:
                    logger.info('Action finished: %s', claim.action.action_id)
                    return
            if batch_failed:
                tools.notification, tools.follow_ups, tools.email, tools.read_email_ids, tools.email_read_performed, tools.management = snapshot
                messages.append({'role': 'system', 'content': 'The last batch failed. Staging from that batch was rolled back; return corrected calls.'})
        raise LLMError('Executor reasoning limit reached; effects were not committed')
