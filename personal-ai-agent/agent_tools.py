"""Allowlisted database tools; user identity is supplied only by the backend."""

from dataclasses import asdict
from pathlib import Path
from copy import deepcopy
from datetime import datetime

from database import connect
from action_data import parse_context
from models import ActionContext, Message
from planning import save_plan, validate_plan
from storage import MessageStore


from tool_definitions import definition

ACTION_SCHEMA = {"type": "object", "additionalProperties": False,
                 "required": ["description", "exec_at", "notification"], "properties": {
                     "description": {"type": "string", "minLength": 1},
                     "exec_at": {"anyOf": [
                         {"type": "object", "additionalProperties": False,
                          "required": ["delay_seconds"], "properties": {
                              "delay_seconds": {"type": "number", "exclusiveMinimum": 0, "maximum": 31536000}}},
                         {"type": "object", "additionalProperties": False,
                          "required": ["at"], "properties": {
                              "at": {"type": "string", "description": "Explicit absolute time requested by user, aware ISO 8601"}}},
                         {"type": "object", "additionalProperties": False,
                          "required": ["immediate"], "properties": {"immediate": {"const": True}}}],
                         "description": "Choose delay_seconds for relative delay, at for absolute schedule, immediate only if requested now"},
                     "notification": {"type": ["string", "null"],
                                      "description": "Exact text for deterministic delivery; null for reasoning actions"},
                 }}
TASK_SCHEMA = {"type": "object", "additionalProperties": False,
               "required": ["description", "period", "actions"], "properties": {
                   "description": {"type": "string", "minLength": 1},
                   "period": {"type": ["integer", "null"], "minimum": 1,
                              "description": "Fixed recurrence in seconds, or null for one-time"},
                   "actions": {"type": "array", "minItems": 1, "maxItems": 1, "items": ACTION_SCHEMA},
               }}
TOOL_DEFINITIONS = [
    definition("respond_chat", "Terminal: reply to conversation or explanation. Never claim to create/change/cancel work here; use apply_plan for that.",
               {"reply": {"type": "string", "minLength": 1}}, ["reply"]),
    definition("ask_clarification", "Terminal: ask for missing details without writing tasks.",
               {"reply": {"type": "string", "minLength": 1},
                "missing_information": {"type": "array", "minItems": 1, "items": {"type": "string", "minLength": 1}}},
               ["reply", "missing_information"]),
    # Read/search tools disabled while all user records are preloaded.
    # definition("find_messages", "Search this user's previous incoming chat messages by keywords. Ranked by word matches, then recency; not semantic search. Use short task keywords and refine as needed.",
    # {"query": {"type": "string", "minLength": 1},
    # "limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["query", "limit"]),
    # definition("get_recent_messages", "Read this user's recent incoming messages before the current message. Useful for short follow-ups such as 'yes', 'it', or 'change that'. Newest first.",
    # {"limit": {"type": "integer", "minimum": 1, "maximum": 20}}, ["limit"]),
    # definition("find_tasks", "Find this user's active tasks by description substring. Use empty query to list all active tasks. Initial active task context is already provided.",
    # {"query": {"type": "string"}}, ["query"]),
    # definition("get_task", "Read one of this user's tasks and its actions, including scheduled times.",
    # {"task_id": {"type": "string"}}, ["task_id"]),
    definition("apply_plan", "Terminal tool: atomically create tasks/actions and cancel old work. For an update, cancel old IDs and create full replacements in this SAME call. Ends the turn on success.",
               {"tasks": {"type": "array", "maxItems": 10, "items": TASK_SCHEMA},
                "cancel_task_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True}},
               ["tasks", "cancel_task_ids"]),
]


class AgentTools:
    def __init__(self, db_path: Path, user_id: str, message: Message) -> None:
        self.path = db_path
        self.user_id = user_id
        self.message = message
        self.seen_active_ids: set[str] = set()
        self.task_refs: dict[str,str] = {}
        self.pending_activity: str | None = None
        self.queue_token: str | None = None
        self.result: ActionContext | None = None

    def execute(self, name: str, arguments: dict, *, require_new_description: str | None = None) -> dict:
        if self.result is not None:
            raise ValueError("A plan was already committed for this message")
        if not isinstance(arguments, dict):
            raise ValueError("Arguments must be an object")
        if name == 'add_task_context':
            if set(arguments) - {'task_id','confirm','reply','information','shared_reference'} or arguments.get('confirm') is not True:
                raise ValueError('add_task_context requires confirm=true, a casual reply, and an optional task_id')
            task_id = arguments.get('task_id')
            if task_id is None and len(self.seen_active_ids) == 1:
                task_id = next(iter(self.seen_active_ids))
            if isinstance(task_id,str):
                task_id = self.task_refs.get(task_id,task_id)
            if task_id not in self.seen_active_ids:
                raise ValueError('Select an ACTIVE task handle from context')
            from task_context import add_task_context
            from planning import nonempty_text
            reply = nonempty_text(arguments.get('reply'))
            information = nonempty_text(arguments.get('information'))
            if information != self.message.message.strip():
                from task_context import ContextUpdateRejected
                raise ContextUpdateRejected('Context information must quote the CURRENT user message verbatim, not a previous message. Do not attach stale facts.')
            count = add_task_context(self.path,self.user_id,task_id,self.message,reply,
                                     arguments.get('shared_reference'),self.queue_token)
            self.result = ActionContext(self.message,[])
            return {'ok': True,'task_id':task_id,'updated_actions':count}
        if name == 'ask_reminder_time':
            self._fields(arguments, {'activity'})
            from planning import nonempty_text
            activity = nonempty_text(arguments['activity'])
            self.result = ActionContext(self.message, ['Reminder time'],
                                       reply=f'When should I remind you to {activity}?', pending_activity=activity)
            return {'ok': True}
        if name == 'set_reminder_time':
            if self.pending_activity is None:
                raise ValueError('There is no pending reminder question')
            if set(arguments) - {'delay','unit','at','period','activity'}:
                raise ValueError('Unexpected time arguments')
            supplied_activity = arguments.get('activity', self.pending_activity)
            if not isinstance(supplied_activity,str) or supplied_activity.strip().casefold() != self.pending_activity.strip().casefold():
                raise ValueError('This tool only completes the pending activity; use create_reminder for different work')
            return self._reminder_tool('create_reminder', {'activity': self.pending_activity, **arguments})
        if name in {'create_reminder', 'update_reminder', 'reschedule_reminder', 'cancel_task'}:
            return self._reminder_tool(name, arguments)
        if name in {"respond_chat", "ask_clarification"}:
            from planning import nonempty_text
            self._fields(arguments, {"reply"} if name == "respond_chat" else {"reply", "missing_information"})
            reply = nonempty_text(arguments['reply'])
            missing = arguments.get('missing_information', [])
            if name == 'ask_clarification':
                if not isinstance(missing, list) or not missing:
                    raise ValueError('Clarification requires missing information')
                missing = [nonempty_text(item) for item in missing]
            self.result = ActionContext(self.message, missing, reply=reply)
            return {"ok": True}
        if name in {"find_messages", "get_recent_messages"}:
            self._fields(arguments, {"query", "limit"} if name == "find_messages" else {"limit"})
            query = arguments.get("query")
            if name == "find_messages" and (not isinstance(query, str) or not query.strip()):
                raise ValueError("query must be non-empty text")
            records = MessageStore(self.path).history(
                self.user_id, self.message.id, arguments["limit"], query=query,
            )
            return {"messages": records}
        if name == "find_tasks":
            self._fields(arguments, {"query"})
            query = arguments["query"]
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            with connect(self.path) as connection:
                tasks = [dict(row) for row in connection.execute(
                    "SELECT * FROM tasks WHERE user_id = ? AND status = 'ACTIVE' "
                    "AND instr(lower(description), lower(?)) > 0 ORDER BY created_at, task_id",
                    (self.user_id, query),
                )]
            self.seen_active_ids.update(task["task_id"] for task in tasks)
            return {"tasks": tasks}
        if name == "get_task":
            self._fields(arguments, {"task_id"})
            if not isinstance(arguments["task_id"], str):
                raise ValueError("task_id must be a string")
            with connect(self.path) as connection:
                row = connection.execute(
                    "SELECT * FROM tasks WHERE task_id = ? AND user_id = ?",
                    (arguments["task_id"], self.user_id),
                ).fetchone()
                if row is None:
                    raise ValueError("Task not found for this user")
                task = dict(row)
                task["actions"] = [dict(row) for row in connection.execute(
                    "SELECT * FROM actions WHERE task_id = ? ORDER BY created_at, action_id",
                    (task["task_id"],),
                )]
            if task["status"] == "ACTIVE":
                self.seen_active_ids.add(task["task_id"])
            return {"task": task}
        if name == "apply_plan":
            self._fields(arguments, {"tasks", "cancel_task_ids"})
            ids = arguments["cancel_task_ids"]
            if not isinstance(ids, list) or not all(isinstance(value, str) for value in ids):
                raise ValueError("cancel_task_ids must be an array of strings")
            if len(ids) != len(set(ids)):
                raise ValueError('Cancellation IDs must be unique')
            for task_id in ids:
                if task_id not in self.seen_active_ids:
                    with connect(self.path) as connection:
                        row = connection.execute('SELECT status FROM tasks WHERE task_id=? AND user_id=?',
                                                 (task_id, self.user_id)).fetchone()
                    if row and row['status'] != 'ACTIVE':
                        raise ValueError(f"Task {task_id} is {row['status']}; it cannot be cancelled or updated. "
                                         "For newly requested work create a new task with no cancellation; "
                                         "do not repeat this cancellation.")
                    raise ValueError('Cancellation IDs must be active owned tasks exposed in context')
            proposals = deepcopy(arguments['tasks'])
            tasks, actions = validate_plan(proposals, self.user_id,
                                           reference_time=datetime.fromisoformat(self.message.timestamp), initial_immediate=True)
            save_plan(self.path, tasks, actions, ids, self.user_id,
                      queue_message_id=self.message.id if self.queue_token else None, queue_token=self.queue_token,
                      require_new_description=require_new_description)
            self.result = ActionContext(self.message, [], tasks, actions, ids)
            return {"ok": True, "tasks": [asdict(task) for task in tasks],
                    "actions": [asdict(action) for action in actions], "cancelled_task_ids": ids}
        raise ValueError(f"Unknown tool: {name}")

    def _reminder_tool(self, name: str, args: dict) -> dict:
        allowed = ({'task_id','confirm'} if name == 'cancel_task' else
                   {'task_id','message'} if name == 'update_reminder' else
                   {'task_id','delay','unit','at','period'} if name == 'reschedule_reminder' else set(REMINDER_FIELDS) | {'activity','allow_duplicate'})
        if name == 'cancel_task' and args.get('confirm', True) is not True:
            raise ValueError('Cancellation must be confirmed')
        if set(args) - allowed:
            raise ValueError('Unexpected reminder arguments')
        if name == 'update_reminder' and 'message' not in args:
            raise ValueError('update_reminder requires the new message; use reschedule_reminder for timing changes')
        task_id = args.get('task_id')
        if isinstance(task_id, str):
            task_id = self.task_refs.get(task_id, task_id)
        if task_id is None and len(self.seen_active_ids) == 1 and name != 'create_reminder':
            task_id = next(iter(self.seen_active_ids))
        task = None
        if name != 'create_reminder':
            if not isinstance(task_id, str) or task_id not in self.seen_active_ids:
                raise ValueError('Choose an ACTIVE task_id from current task context; completed tasks cannot be updated or cancelled')
            with connect(self.path) as db:
                row = db.execute("SELECT * FROM tasks WHERE task_id=? AND user_id=? AND status='ACTIVE'", (task_id, self.user_id)).fetchone()
                if row is None:
                    raise ValueError('Task already completed or cancelled. Do not change it; ask whether new work is wanted')
                task = dict(row)
                pending = db.execute("SELECT * FROM actions WHERE task_id=? AND status IN ('PENDING','RUNNING') ORDER BY exec_at DESC LIMIT 1", (task_id,)).fetchone()
            if name == 'cancel_task':
                return self.execute('apply_plan', {'tasks': [], 'cancel_task_ids': [task_id]})
        from planning import nonempty_text
        from scheduling import MissingScheduleError
        text = args.get('message')
        if name == 'create_reminder' and 'activity' in args:
            from planning import nonempty_text
            text = 'Reminder: ' + nonempty_text(args['activity']).rstrip('.!') + '.'
        if text is None and task:
            text = (parse_context(pending['context']).get('legacy_message') if pending else None) or task['description']
        text = nonempty_text(text)
        if 'at' in args:
            if 'delay' in args or 'unit' in args:
                raise ValueError('Choose at OR delay+unit')
            schedule = {'at': args['at']}
        elif 'delay' in args or 'unit' in args:
            amount, unit = args.get('delay'), args.get('unit')
            factors = {'seconds': 1, 'minutes': 60, 'hours': 3600, 'days': 86400}
            if type(amount) not in (int, float) or amount <= 0 or unit not in factors:
                raise MissingScheduleError('Ask the user for the reminder time; delay and unit must be supplied together and positive')
            schedule = {'delay_seconds': amount * factors[unit]}
        elif task and pending:
            schedule = {'at': parse_context(pending['context']).get('requested_time') or pending['exec_at']}
        else:
            raise MissingScheduleError('Reminder time is missing. Ask when to remind the user.')
        if name == 'create_reminder':
            allow_duplicate = args.get('allow_duplicate', False)
            if type(allow_duplicate) is not bool:
                raise ValueError('allow_duplicate must be a boolean')
        proposal = {'description': text, 'period': args.get('period', task['period'] if task else None),
                    'actions': [{'exec_at': schedule, 'description': text, 'notification': text}]}
        return self.execute('apply_plan', {'tasks': [proposal], 'cancel_task_ids': [task_id] if task else []},
                            require_new_description=text if name == 'create_reminder' and not allow_duplicate else None)

    @staticmethod
    def _fields(arguments: dict, expected: set[str]) -> None:
        if set(arguments) != expected:
            raise ValueError("Unexpected or missing tool arguments")

# Small native tools: the model supplies the amount/unit; Python does arithmetic.
REMINDER_FIELDS = {
    'message': {'type': 'string', 'description': 'Reminder text about the activity, without repeating the delay'},
    'delay': {'type': 'number', 'description': 'Requested amount, e.g. 5 for five minutes'},
    'unit': {'type': 'string', 'enum': ['seconds', 'minutes', 'hours', 'days']},
    'at': {'type': 'string', 'description': 'Only for an absolute date/time: timezone-aware ISO 8601'},
    'period': {'type': ['integer', 'null'], 'description': 'Recurring interval in seconds; omit/null for one time'},
}
PLANNER_TOOLS = [
    definition('create_reminder', 'Create a NEW reminder, leaving every existing reminder unchanged. Also adding another activity uses this tool. Supply activity and delay+unit OR at.',
               {**{k:v for k,v in REMINDER_FIELDS.items() if k != 'message'},
                'activity': {'type': 'string', 'description': 'Only the activity, e.g. drink water. Do not include timing or a promise.'},
                'allow_duplicate': {'type':'boolean','description':'Only true if the user explicitly requests an additional identical reminder; otherwise omit/false'}}, ['activity']),
    definition('add_task_context', 'Attach the CURRENT user message verbatim as supplementary information to an ACTIVE reminder. Adds description and a notification note, preserves timing, and creates no task. Only use when the information directly changes how this specific action should be carried out. Broad topic similarity is insufficient. Supply a short casual reply to the user, without mentioning internal updates or scheduling.',
               {'task_id': {'type':'string'}, 'confirm': {'type':'boolean','const':True},
                'reply': {'type':'string','minLength':1,'maxLength':600,'description':'Address the user naturally; never speak as if their personal facts belong to you'},
                'information': {'type':'string','minLength':1,'description':'Exact CURRENT user message, quoted verbatim. Never copy a previous user message.'},
                'shared_reference': {'type':'string','minLength':3,'description':'Specific item/person/place explicitly present in BOTH current message and existing action; not a generic category, pronoun or verb. Without one, reply as chat.'}}, ['task_id','confirm','reply','information','shared_reference']),
    definition('update_reminder', 'Change ONLY the text/description of an ACTIVE reminder. Its existing delivery time is always preserved. For timing changes use reschedule_reminder.',
               {'task_id': {'type': 'string'}, 'message': REMINDER_FIELDS['message']}, ['task_id', 'message']),
    definition('reschedule_reminder', 'Change ONLY the time of an ACTIVE reminder. Supply delay+unit or at. Its reminder text is preserved.',
               {'task_id': {'type': 'string'}, **{k:v for k,v in REMINDER_FIELDS.items() if k != 'message'}}, ['task_id']),
    definition('cancel_task', 'Cancel an ACTIVE task and its outstanding actions.',
               {'task_id': {'type': 'string'}, 'confirm': {'type': 'boolean', 'const': True}}, ['task_id','confirm']),
    definition('ask_reminder_time', 'A reminder was requested but no time was supplied. Save the activity and ask the user when.',
               {'activity': {'type': 'string', 'description': 'The requested activity, e.g. drink water'}}, ['activity']),
]

PENDING_TIME_TOOL = definition('set_reminder_time', 'Complete the pending reminder by supplying its time. The activity is already known. Use for answers such as in two minutes.',
    {**{key:value for key,value in REMINDER_FIELDS.items() if key != 'message'},
     'activity': {'type':'string','description':'Optional repeat of the already-bound activity; cannot change it'}}, [])
