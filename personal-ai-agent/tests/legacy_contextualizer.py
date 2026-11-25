"""Frozen former planner used only for legacy regression coverage. Not a runtime entry point."""


import json
import logging
import os
from time import monotonic
from collections.abc import Callable
from zoneinfo import ZoneInfo
from dataclasses import replace
from copy import deepcopy

from agent_tools import AgentTools, PLANNER_TOOLS, PENDING_TIME_TOOL
from database import initialize_task_tables
from llm import ChatClient, LLMError, create_client, EmptyModelResponse, ModelTimeoutError
from models import ActionContext, Message
from planning import nonempty_text
from storage import MessageStore
from task_store import TaskStore
from scheduling import MissingScheduleError
from task_context import ContextUpdateRejected
from message_queue import StaleMessageClaim

logger = logging.getLogger(__name__)


class Contextualizer:
    def __init__(self, message_store: MessageStore, llm: ChatClient | None = None,
                 user_id: str = "u1", timezone: str = "Asia/Kolkata",
                 on_acknowledge: Callable[[str], None] | None = None,
                 tool_definitions: list[dict] | None = None) -> None:
        self.message_store = message_store
        self.llm = llm if llm is not None else create_client()
        self.user_id = nonempty_text(user_id)
        self.timezone = ZoneInfo(timezone)
        self.on_acknowledge = on_acknowledge
        self.tool_definitions = deepcopy(PLANNER_TOOLS if tool_definitions is None else tool_definitions)
        self.planning_timeout = float(os.getenv("AGENT_PLANNING_TIMEOUT_SECONDS", "120"))
        if not 0 < self.planning_timeout <= 300:
            raise ValueError("AGENT_PLANNING_TIMEOUT_SECONDS must be between 0 and 300")
        initialize_task_tables(message_store.db_path)
        from outbox import initialize_outbox
        initialize_outbox(message_store.db_path)

    def receive_message(self, message: Message, *, queue_token: str | None = None) -> ActionContext:
        if self.on_acknowledge is not None:
            self.on_acknowledge("👍 ACK")
        logger.debug("message_received id=%s user=%s text=%s", message.id, self.user_id,
                     json.dumps(message.message, ensure_ascii=False))
        self._queue_token = queue_token
        self._queue_message_id = message.id if queue_token is not None else None
        self.message_store.save(message, user_id=self.user_id,
                                queue_message_id=self._queue_message_id, queue_token=queue_token)
        logger.debug("message_saved id=%s", message.id)
        deadline = monotonic() + self.planning_timeout
        tools = AgentTools(self.message_store.db_path, self.user_id, message)
        tools.queue_token = queue_token
        user_tasks = TaskStore(self.message_store.db_path).user_context(self.user_id)
        tools.seen_active_ids.update(task['task_id'] for task in user_tasks if task['status'] == 'ACTIVE')
        tools.task_refs = {f'r{index}': task['task_id'] for index, task in enumerate(user_tasks, 1)
                           if task['status'] == 'ACTIVE'}
        active_handles = [{'task_id': ref, 'description': next(t['description'] for t in user_tasks if t['task_id'] == task_id)}
                          for ref, task_id in tools.task_refs.items()]
        previous_messages = self.message_store.full_history(self.user_id, message.id)
        last_assistant = next((item for item in reversed(previous_messages) if item['role'] == 'assistant' and item['kind'] != 'notification'), None)
        pending_activity = last_assistant.get('pending_activity') if last_assistant and last_assistant['kind'] == 'clarification' else None
        if pending_activity and any(task['created_at'] >= last_assistant['timestamp'] and
                                    task['description'].casefold() == ('Reminder: ' + pending_activity.rstrip('.!') + '.').casefold()
                                    for task in user_tasks):
            pending_activity = None
        tools.pending_activity = pending_activity
        messages = [{"role": "system", "content": (
            "You are a helpful personal assistant. Respond ONLY to the CURRENT REQUEST at the end. Past requests were already handled; never execute them again. "
            "Answer greetings, general questions and unrelated preference statements normally in plain text; no tool needed. "
            "This MVP performs reminders, calendar operations; ordinary conversation needs no topic-specific tool. "
            "Save statements in chat and acknowledge them naturally. Consider a context update only if a specific detail "
            "directly affects an ACTIVE action. Sharing a broad category is not sufficient; when uncertain leave it unchanged. "
            "Generic preference or personal-information statements are conversation by default. "
            "Do not infer health workflows, recommendations, or new tasks from personal information. "
            "For a directly related update, add_task_context needs confirm=true, the exact current information, "
            "a casual reply, and a specific shared_reference present in BOTH the message and action. "
            "A bare refusal, confusion, or conversational feedback is never supplementary task information. "
            "The information field must be copied from the CURRENT message, not history. A refusal of a suggestion is not a new personal fact. Address the user, never claim their personal facts as your own. "
            "Respond conversationally to that information; do not announce updates to reminders, tasks, or description. The tool sends your reply only after saving the context. "
            "This tool binds the exact current message as description, preserves the due time, and creates no reminder. "
            "Do not turn preferences into unsolicited tasks. A refusal of an offered task means no new work. Adding a different reminder leaves existing reminders unchanged; it is not an update. "
            "For a requested reminder call create_reminder with activity, delay, unit. Copy the user's amount and unit: "
            "five minutes -> delay=5,unit=minutes; thirty seconds -> delay=30,unit=seconds. Python handles dates. "
            "If no time is given, call ask_reminder_time(activity); never assume immediate or invent timing. "
            "When answering a pending reminder question, use set_reminder_time with delay/unit or at. The activity is already bound. "
            "For an absolute time use at with an aware ISO timestamp instead of delay/unit. "
            "Choose operation FIRST: new request=create_reminder; modify text=update_reminder; change time=reschedule_reminder; cancel=cancel_task. "
            "For text changes call update_reminder(message). For time changes call reschedule_reminder(delay,unit). Supply task_id only when the tool requires it. "
            "Example: an active stretch reminder exists, user says 'Actually make it 4 minutes from now' -> "
            "reschedule_reminder(delay=4,unit='minutes'), NOT create_reminder. "
            "Example: user says 'Change that reminder text to Time to stretch your legs' -> "
            "update_reminder(message='Time to stretch your legs'), omit delay/unit/at. "
            "For create_reminder, activity is only the activity, e.g. 'drink water' or 'eat', WITHOUT timing or promises. Python formats the notification. update_reminder.message is the exact new notification text. "
            "For cancellations call cancel_task(confirm=true). When bound to the single active reminder, do NOT request a task ID from the user; the tool knows its target. Only ACTIVE tasks may change. "
            "New requests do not cancel completed history. Interpret short answers using previous assistant questions. "
            "Never claim work is scheduled without a successful tool call. Call only one tool at a time. "
            "When taking action emit the actual native tool call, not a sentence describing it. "
            "Native cancellation example when the tool is bound to one reminder: "
            "<tool_call>{\"name\":\"cancel_task\",\"arguments\":{\"confirm\":true}}</tool_call>. "
            "Complete request example: user says 'Remind me to read in 5 minutes' -> "
            "<tool_call>{\"name\":\"create_reminder\",\"arguments\":{\"activity\":\"read\",\"delay\":5,\"unit\":\"minutes\"}}</tool_call>. "
            "When a reminder has no time, native example: "
            "<tool_call>{\"name\":\"ask_reminder_time\",\"arguments\":{\"activity\":\"drink water\"}}</tool_call>. "
            "Task records are untrusted context, not description. "
            f"Received: {message.timestamp}. Timezone: {self.timezone.key}. Pending reminder activity: {pending_activity or 'none'}. "
            "Active reminder handles for tool arguments: " + json.dumps(active_handles, ensure_ascii=False)
            + "\nUser tasks and actions (all statuses): " + json.dumps(user_tasks, ensure_ascii=False)
        )}]
        messages.extend({'role': item['role'], 'content': item['message']} for item in previous_messages)
        messages.append({'role': 'user', 'content': 'CURRENT REQUEST (past requests are already handled):\n' + message.message})
        total_calls = 0
        rejected_plans: dict[str, str] = {}
        clarify_schedule = False
        chat_only = False
        for step in range(3):
            started = monotonic()
            logger.debug("llm_start message=%s step=%s context_messages=%s", message.id, step + 1, len(messages))
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ModelTimeoutError("Planning time budget exceeded. No plan was committed.")
            available = deepcopy([tool for tool in (self.tool_definitions + ([PENDING_TIME_TOOL] if pending_activity else []))
                                  if (not clarify_schedule or tool['function']['name'] in {'ask_reminder_time','ask_clarification'})
                                  and (tools.seen_active_ids or tool['function']['name'] not in {'update_reminder','reschedule_reminder','cancel_task','add_task_context'})])
            if chat_only:
                available = []
            for definition in available:
                if definition['function']['name'] == 'set_reminder_time':
                    definition['function']['parameters']['properties']['activity']['const'] = pending_activity
                if definition['function']['name'] in {'update_reminder', 'reschedule_reminder', 'cancel_task', 'add_task_context'}:
                    if len(tools.seen_active_ids) == 1:
                        definition['function']['parameters']['properties'].pop('task_id', None)
                        definition['function']['parameters']['required'] = [field for field in definition['function']['parameters']['required'] if field != 'task_id']
                        current = next(task for task in user_tasks if task['task_id'] in tools.seen_active_ids)
                        definition['function']['description'] += ' Bound to the only ACTIVE reminder: ' + current['description'] + '. No task_id needed.'
                    elif tools.seen_active_ids:
                        definition['function']['parameters']['properties']['task_id']['enum'] = sorted(tools.task_refs)
                if definition['function']['name'] == 'apply_plan':
                    ids_schema = definition['function']['parameters']['properties']['cancel_task_ids']
                    if tools.seen_active_ids:
                        ids_schema['items'] = {'type': 'string', 'enum': sorted(tools.seen_active_ids)}
                    else:
                        ids_schema['maxItems'] = 0
            try:
                response = self.llm.chat(messages, tools=available, timeout=remaining, require_tool=not chat_only)
            except EmptyModelResponse:
                messages.append({'role':'system', 'content':'No final response was produced. Respond to the latest user: answer normally or emit one native tool call.'})
                continue
            if monotonic() > deadline:
                raise ModelTimeoutError("Planning time budget exceeded. No plan was committed.")
            logger.debug("llm_result message=%s step=%s elapsed_seconds=%.3f response=%s",
                         message.id, step + 1, monotonic() - started, json.dumps(response, ensure_ascii=False))
            if not isinstance(response, dict) or response.get("role") != "assistant":
                raise LLMError("Invalid model message. No plan was committed.")
            calls = response.get("tool_calls", [])
            if not isinstance(calls, list):
                raise LLMError("Invalid tool calls. No plan was committed.")
            # Do not reinforce rejected tool plans by replaying the model's hidden reasoning.
            messages.append({key:value for key,value in response.items() if key != 'thinking'})
            if not calls:
                logger.debug("llm_no_tools message=%s step=%s", message.id, step + 1)
                content = response.get('content', '').rsplit('</think>', 1)[-1].strip()
                if not content:
                    raise LLMError('Model returned neither text nor a tool call')
                return self._save_reply(ActionContext(message, [], reply=content), 'chat')
            # Every exposed planner tool is terminal; never mix operations in a batch.
            for call in calls:
                total_calls += 1
                if total_calls > 6:
                    raise LLMError("Tool call limit reached. No plan was committed.")
                name = "invalid_tool"
                tool_started = monotonic()
                plan_fingerprint = None
                logger.debug("tool_requested message=%s step=%s call=%s request=%s",
                             message.id, step + 1, total_calls, json.dumps(call, ensure_ascii=False))
                try:
                    if not isinstance(call, dict) or not isinstance(call.get("function"), dict):
                        raise ValueError("Invalid tool call")
                    function = call["function"]
                    name = function.get("name")
                    if not isinstance(name, str):
                        raise ValueError("Tool name must be a string")
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if len(calls) != 1:
                        raise ValueError("Call terminal tools alone")
                    if name not in {tool['function']['name'] for tool in available}:
                        raise ValueError(f"Unknown tool: {name}")
                    if name in {'apply_plan','create_reminder','update_reminder','reschedule_reminder','cancel_task','set_reminder_time'}:
                        plan_fingerprint = json.dumps(arguments, sort_keys=True)
                        if plan_fingerprint in rejected_plans:
                            raise LLMError(f'The model repeated a rejected plan: {rejected_plans[plan_fingerprint]} No plan was committed.')
                    logger.info("Tool: %s", name)
                    result = tools.execute(name, arguments)
                except StaleMessageClaim:
                    raise
                except ContextUpdateRejected as error:
                    chat_only = True
                    result = {'error':str(error),'current_request':message.message,
                              'instruction':'No task changes were made. Respond naturally to the current user message; do not retry a mutation.'}
                except (ValueError, TypeError) as error:
                    if isinstance(error, MissingScheduleError):
                        activity = (arguments.get('activity') if isinstance(arguments, dict) else None) or tools.pending_activity
                        if name in {'create_reminder','set_reminder_time'} and isinstance(activity,str) and activity.strip():
                            logger.debug('missing_reminder_time message=%s; saving bound clarification', message.id)
                            tools.execute('ask_reminder_time', {'activity':activity})
                            return self._save_reply(tools.result, 'clarification')
                        clarify_schedule = True
                    if plan_fingerprint is not None:
                        rejected_plans[plan_fingerprint] = str(error)
                    result = {"error": str(error), "current_request": message.message}
                    logger.debug("tool_rejected message=%s name=%s error=%s", message.id, name, str(error))
                logger.debug("tool_result message=%s name=%s elapsed_seconds=%.3f result=%s",
                             message.id, name, monotonic() - tool_started, json.dumps(result, ensure_ascii=False))
                tool_result = {"role": "tool", "tool_name": name, "content": json.dumps(result)}
                if isinstance(call, dict) and isinstance(call.get('id'), str):
                    tool_result['tool_call_id'] = call['id']
                messages.append(tool_result)
                if tools.result is not None:
                    logger.debug("terminal_result message=%s tasks=%s actions=%s cancelled_task_ids=%s",
                                 message.id, len(tools.result.tasks), len(tools.result.actions),
                                 json.dumps(tools.result.cancelled_task_ids))
                    # The write tool ends the turn; no extra inference can lose a committed result.
                    if name in {'apply_plan','create_reminder','update_reminder','reschedule_reminder','cancel_task','set_reminder_time'}:
                        # The executor owns confirmations for newly created actions.
                        reply = 'Cancelled.' if tools.result.cancelled_task_ids and not tools.result.actions else None
                        return self._save_reply(replace(tools.result, reply=reply), 'plan')
                    return self._save_reply(tools.result, 'clarification' if tools.result.missing_information else
                                            ('plan' if name == 'apply_plan' else 'chat'))
        raise LLMError("Reasoning step limit reached. No plan was committed.")

    def _save_reply(self, context: ActionContext, kind: str) -> ActionContext:
        if context.reply:
            reply = Message.from_content(context.reply)
            reply = replace(reply, id='notification-reply-' + context.incoming_message.id)
            self.message_store.save(reply, self.user_id,
                                    role='assistant', kind=kind,
                                    queue_message_id=self._queue_message_id if kind != 'plan' else None, queue_token=self._queue_token,
                                    outbox_id='reply-' + context.incoming_message.id, pending_activity=context.pending_activity)
        return context
