# Personal AI Agent

Terminal assistant using DeepSeek (default), optional local Ollama, native tools, and SQLite. Current user-facing
capabilities are conversation and reminders, including clarification, rescheduling,
text changes, cancellation, scheduled delivery, and continuous/proactive processing.
External application integrations will be added later.

## Run

DeepSeek V4.1 Flash uses the API model ID `deepseek-flash`.
Set your key privately in zsh, then start the app:

```zsh
read -s 'DEEPSEEK_API_KEY?DeepSeek API key: '; echo
export DEEPSEEK_API_KEY
python3 main.py
```

The app reads `DEEPSEEK_API_KEY` from the environment. Do not paste it into code,
logs, or chat. `.env` files are not automatically loaded. Both planner and executor
use this provider. Chat history and supplied task/action context are sent to DeepSeek;
SQLite remains local. Live API verification requires your configured key.

To return to local inference:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen3:0.6b python3 main.py
```

Submit messages with Enter. Input stays responsive while a background worker
processes messages in order. Each submission receives `👍 ACK`. For example:

```text
You: Remind me to drink water in 1 minute
Agent: 👍 ACK
Agent: Scheduled the reminder.
... at the requested time ...
Agent: Reminder: drink water.
```

Keep the process running to see notifications. Ctrl+C stops the application;
unprocessed messages remain durable. The shutdown message reports actual pending
or running message counts, not failed historical requests.

## Separate controller

Normally `main.py` includes the action controller. To run action execution in a
separate process:

```bash
# Terminal A
python3 controller.py

# Terminal B
AGENT_WORKER_ENABLED=0 python3 main.py
```

The standalone controller executes actions but does not consume outgoing messages.
Only the chat process drains the outbox, even with embedded action execution
disabled. Notifications wait in SQLite while chat is closed. Restart BOTH
processes after code changes. Multiple simultaneous chat processes for the same
user are not a supported delivery configuration.

## Processing flow

1. Main durably queues incoming messages and `👍 ACK` in FIFO order.
2. Contextualizer retrieves user-scoped history, tasks and pending actions, classifies
   the message, and enqueues exactly one immediate action. For new requests only,
   it atomically creates the initial task record and links that kickoff action.
3. Contextualizer never edits existing tasks/actions or replies directly. Conversation,
   additional information, updates, cancellation and answers all go through Executor.
4. Action Executor owns subsequent task/action management, operations and communication.
   For a future request it acknowledges now and schedules one next action for later.
5. The dispatcher claims due actions. Executor evaluates the latest description/context
   and uses tools; main delivers committed outgoing messages from the shared outbox.

## Agent tools and ownership

Contextualizer exposes only `create_action(task_id, description, context, message_type,
new_task?)`. `new_task` holds the initial description and optional recurrence period;
all other message types identify an existing task or use a null task ID. The backend
owns IDs, user identity, timestamps and the exact original request/receipt in context.
Every accepted message produces exactly one action at NOW, with an idempotent receipt.

Executor exposes:

| Tools | Responsibility |
| --- | --- |
| `read_tasks`, `read_actions` | Inspect owned task state and action context. |
| `select_task`, `create_task` | Resolve an unbound task request; ordinary new tasks already exist. Generic chat cannot create tasks. |
| `update_task`, `cancel_task` | Change the associated task or cancel its outstanding work. |
| `update_action`, `cancel_action` | Replace or invalidate affected pending/running actions. A replacement fences stale workers; omitted timing preserves exec_at. |
| `add_actions` | Stage at most one next follow-up with description, structured context and execution time. |
| `send_message` | Stage user communication in the outgoing queue. |
| `draft_email`, `send_email`, `read_email` | Work with the local demo mailbox. |
| `finish_action` | Commit verified effects, management changes, outcome and optional task status atomically. |

There is no literal shared-phrase validation. Executor evaluates conversational
references and semantic relevance; context merges preserve earlier constraints.
The tools validate ownership, allowed fields, state transitions and concurrent changes.
All effects and outgoing success messages commit together; failed/stale batches publish
no success. Executor has six working model turns plus one completion-only turn and
sixteen tool calls, so sequential read/update/message calls can reach completion.

COMPLETED means the current action finished. Executor must explicitly mark a task
COMPLETED only after its goal is achieved; pending operational work prevents that.
WAITING records missing information and keeps a task ACTIVE. Metadata changes and
acknowledgments do not reset the recurrence clock. The Continuous User Processor
alone generates recurring occurrences through the incoming queue. Proactive remains
separate: read-only observations, suggestions through the outbox, and immediate
actions for existing tasks; no execution or independent task creation.

## Configuration

`.env.example` is a reference; this app does not automatically load `.env` files.
Set environment variables in your shell.

| Variable | Default |
| --- | --- |
| `LLM_PROVIDER` | `deepseek` |
| `DEEPSEEK_MODEL` | `deepseek-flash` |
| `DEEPSEEK_API_KEY` | required for DeepSeek |
| `OLLAMA_BASE_URL` | `http://localhost:11434` |
| `OLLAMA_MODEL` | `qwen3:0.6b` |
| `OLLAMA_TOOL_MODE` | `native` |
| `OLLAMA_THINK` | `1` for Qwen3 0.6B, otherwise `0` |
| `OLLAMA_NUM_PREDICT` | `1024` |
| `OLLAMA_NUM_CTX` | `8192` |
| `OLLAMA_REQUEST_TIMEOUT_SECONDS` | `120` |
| `AGENT_PLANNING_TIMEOUT_SECONDS` | `120`, shared across three model turns |
| `AGENT_USER_ID` | `u1` |
| `AGENT_TIMEZONE` | `Asia/Kolkata` |
| `AGENT_WORKER_ENABLED` | `1` |
| `AGENT_DB_PATH` | Project's `data/agent.db` |

Qwen3 0.6B native tool selection was more reliable with thinking enabled in live
checks. Reasoning is not displayed or parsed as executable tools. The legacy
structured transport is opt-in and is not used for current live verification.

A restart uses the same user/database. No new user is created. Different configured
user IDs have separate context, tasks, queue claims, and outgoing messages.

## Storage, retries, and privacy

`data/agent.db` contains `messages`, `message_inbox`, `tasks`, `actions`,
`action_leases`, `notifications`, `planner_commits`, and
`action_sources`, `action_results`, `continuous_users`, `proactive_seen`, `proactive_effects`, and
`recurring_triggers`. `notifications` is the shared
outgoing buffer; its action ID is nullable for ordinary chat/acknowledgments.
Delivered replies enter assistant history once; acknowledgments do not pollute
conversation history. Existing schemas migrate without discarding records.

Message states are PENDING/RUNNING/DONE/FAILED. Connectivity/timeouts retry up to
three attempts with five-second backoff. Later messages wait behind a retry;
failed messages remain stored but no longer block subsequent work. Interrupted
RUNNING messages recover after their lease expires (planning budget + 60 seconds).
Plan commits mark the inbox DONE in the same transaction, preventing replay after
a crash following commit. Claim tokens reject stale workers. Terminal delivery
is at least once: a crash between printing and recording delivery can repeat a
notification. Normal process exits preserve pending outbox data.

DeepSeek is the default and receives supplied chat/task context. Inference stays
local only when `LLM_PROVIDER=ollama` points to a local Ollama server. Database and debug logs
contain private message text and tool arguments. `data/`, `tmp/`, `.env`, and
Python caches are ignored by Git. Logs are in `tmp/agent.log`; routine polling and
tool logs do not print over terminal input. Full context grows with use; compaction
and model quality remain limitations. Hardware/OS restarts require separate
diagnosis; selecting a smaller model does not establish their cause.

## Verification

Backend regression tests do not call a model:

```bash
python3 -m unittest discover -s tests -q
```

Opt-in real model scenarios run against temporary databases:

```bash
python3 tests/live_agent.py
```

The optional runner checks chat, immediate task/kickoff creation, executor scheduling,
unrelated information, and cancellation. It does not wait for real-time delivery.
It makes model/API calls only when explicitly launched; no live check was run for
this update. Model responses in backend tests are controlled fixtures and do not
prove live conversational reliability.

## Proactive Agent and Continuous User Processor

`main.py` starts the continuous processor in the background. It discovers stored
users and checks recurring ACTIVE tasks every five seconds. A due task (period
elapsed since last execution, with no outstanding action or incoming trigger)
gets one internal incoming message for the existing Contextualizer worker.
Trigger insertion and its deduplication receipt commit together. The task keeps
its ID. Executor completion does not independently generate another occurrence.

Continuous and proactive processing are enabled by default:

```zsh
python3 main.py
```

Local read-only `email_inbox` and `reminders` sources are connected. Newly registered
users receive these local source grants; saved grants and revocations survive restart.
The Proactive Agent skips inference when permitted snapshots have no new changes. Set `AGENT_PROACTIVE_ENABLED=0` to disable observation, or
`AGENT_PROACTIVE_INTERVAL_SECONDS` to change the interval (default 30 seconds).
The controller polls recurring tasks every five seconds by default.

Future read-only providers implement `snapshot(user_id)` and return
`Observation(source, object_id, data, task_id?)`. Register them with `ProactiveAgent`
and grant access per user through `configure_user`. Calendar tests use injected
fixtures, not a real account. Stable IDs and dated/versioned data let fingerprints
detect changes. No real external email/account adapter is included; the mailbox source reads SQLite only.

The agent receives new permitted snapshots, full user chat history, tasks and action contexts, and
read-only tools (`read_email`, `read_actions`, `read_user_context`, `read_observation`, `get_task_context`). Application functions
`send_message` and `add_actions` stage effects; `finish_check` commits
them atomically. A changed snapshot can be processed in one model call; unchanged
snapshots make none. A pass processes at most twenty changes, three model calls,
and eight tools. Permissions/leases are rechecked before reads, inference, and
commit. Model calls use the existing configured provider, so opting into DeepSeek
observation sends that permitted data to DeepSeek.

New opportunities go straight to the shared outgoing buffer as suggestions or
questions, with no automatic task. Replies enter the normal terminal inbox.
Relevant changes create immediate actions on existing ACTIVE owned tasks. External
content is explicitly untrusted in both observation and executor contexts; these
controls reduce prompt-injection exposure but do not prove model semantic safety.
No external mutation tools are exposed to the Proactive Agent.

Fingerprints and effect receipts prevent repeat processing, notifications, or
actions for the same observation version, including after restarts. Raw snapshots
are not persisted by the observer; relevant excerpts included in executor actions
and user messages are stored locally as part of the normal workflow. Changing or
revoking source permissions invalidates in-flight checks. Failed checks leave
observations eligible for the next interval.

For multiple users, call `configure_user(user_id, proactive_enabled=..., sources=...,
interval_seconds=..., recurrence_enabled=...)` on the continuous processor. Settings
are persisted in SQLite. Observation is enabled for the current user at startup unless
`AGENT_PROACTIVE_ENABLED=0` is set. The default runtime has local mailbox and task-schedule sources. No Gmail or calendar account is connected.
Directly configured source grants apply only to explicitly registered adapters. To use a standalone continuous processor:

```bash
python3 continuous_user_processor.py
```

The normal message worker and action controller must also be running. Multiple
continuous processes are fenced by per-user leases and durable trigger receipts.
There is no separate service, dependency, or framework to install.

## Action description, context, and execution history

Actions have `description` (the step) and `context` (a JSON object of facts,
preferences, constraints and task history). The former `instructions` field is
removed. Executor decides how to perform the step. SQLite stores context as JSON
text; models, tools and Python action objects receive a parsed object, never a
JSON-encoded string. Incoming context tool arguments must be objects.

`context.past_actions` contains every past task action once, plus the current
committed action when a follow-up is created. Each record has its identity,
description, timing, state/outcome, factual context, completion reason and verified
tool results. History entries do not contain another nested `past_actions` list.
The backend derives history from task records; model-supplied invented history is
ignored. Nested decisions/preferences merge so unrelated earlier facts survive.
The executor receives this context and current task state without full conversation
history. All past actions are retained; no compaction or truncation is applied.

Planner actions always start NOW. The backend preserves the original request
and receipt timestamp in `context.request`; the planner does not resolve execution
timestamps. Executor interprets timing and calls `add_actions` with a description,
structured context, and `exec_at`. Relative kickoff delays use the request receipt;
subsequent continuation delays use the current execution time. A timing change
creates an immediate Contextualizer handoff; Executor updates or replaces the existing pending action.

Actions have no `scheduled_for` or prepared `notification` fields. Every due action
runs through the executor, including reminder delivery. Existing databases migrate
legacy timing/message facts into context before removing those columns, retaining
IDs, history, and related queue records.

## Local email mailbox (no external delivery)

The executor has `draft_email`, `send_email`, and `read_email`. Email tools operate on SQLite
only; there is no SMTP, Gmail, or real delivery. The model chooses tools from the
task context; no email keyword routing or document-access workflow is hard-coded.
Missing recipient addresses or content should produce a question, not invented mail.

Each `emails` record has `user_id`, `thread_id`, `email_id`, `timestamp`, `from`,
`to`, `subject`, `content`, `status` (DRAFT/SENT/RECEIVED), and boolean `read`. IDs, sender, owner, and UTC timestamp are
backend-assigned. An internal sequence breaks timestamp ties, and an internal
unique action link prevents an action from sending twice. Replies append records
with the same thread ID rather than overwriting previous messages. SQLite encodes
read flags as 0/1; tool responses expose true/false.

- `send_email(to, subject, content, thread_id?)`: null thread ID starts a thread; an existing
  owned thread ID appends a reply. One outgoing mail per action is staged and
  committed atomically with `finish_action`, task state, history, and optional
  user-facing confirmation. Cancellation, stale claims, or a rejected batch do not
  send an email. Sent records start `read=true`.
- `read_email(thread_id?, email_id?, limit?, unread_only?, mark_read?)`: inspect
  only the current user's mailbox, ordered oldest-first within the result. The
  default limit is 50, maximum 100. Neither read flag changes by default.
  `mark_read=true` stages the selected read updates until `finish_action`; readonly
  observation can use `unread_only=true` without consuming messages.

Optionally set the current user's local sender address before starting:

```zsh
export AGENT_EMAIL_ADDRESS='me@local.invalid'
python3 main.py
```

Without an override, the backend generates a stable per-user `@local.invalid`
address. The model cannot choose another sender or mailbox user. To inspect the
local records or simulate an incoming reply:

```bash
python3 email_store.py list
python3 email_store.py list --thread THREAD_ID
python3 email_store.py receive --thread THREAD_ID --from colleague@example.test --content 'Access granted.'
```

Simulated incoming records start unread. The CLI respects `AGENT_USER_ID` and
`AGENT_DB_PATH`; `--user` can select a local test mailbox. The application tools
cannot use that override. Content is untrusted data. The local mailbox is monitored by the permission-scoped Proactive Agent;
observation receipts track processed messages without changing read flags.

### Drafts and subject handling

`draft_email(to?, subject?, content, thread_id?)` saves an **unsent** DRAFT. Recipient
and subject may be completed later; content is required. `draft_email(email_id, ...)`
can edit that owned draft while keeping its IDs. Drafting does not send a message.

`send_email(email_id=...)` sends that DRAFT in the local mailbox, retaining its
email ID and thread ID. Optional `to`, `subject`, and `content` update it before
sending. Alternatively, create a new SENT record by supplying recipient, subject,
and content without an email ID. Sending requires a valid address and nonempty
subject/body. The model can choose a sensible subject when the user delegates it;
it is now a stored field, not merely text in the confirmation.

SENT and RECEIVED records cannot be edited or sent again by ID. Asking to change
an already-sent message should lead to clarification about an additional message,
not a silent duplicate send. An intentional additional email is a separate record.
State-based checks prevent resending the same draft; semantic similarity between
separate new-send requests is still the agent's responsibility.

Email effects remain staged until action completion. Draft edits/sends use revision
checks, so a stale action cannot overwrite an edited draft or send it twice.
Verified history records `draft_email` or `send_email` with the committed state,
subject, email ID and thread ID. Contextualizer receives these receipts along
with explicit recent dialogue to resolve short answers against the current topic.
Backend tests verify data and protocol behavior, not all live language decisions.

`read_email(status='DRAFT')` lists drafts; SENT and RECEIVED filters work similarly.
The CLI supports the same filter:

```bash
python3 email_store.py list --status DRAFT
python3 email_store.py list --status SENT
```

Existing committed outgoing mail migrates as SENT, never DRAFT; prior subjects
that were never stored remain blank. Existing simulated incoming mail migrates as
RECEIVED. These state labels describe the local mailbox only; there is still no
external email delivery. Local proactive mailbox monitoring is connected.


## Local proactive monitoring

The supplied proactive prompt is in `prompts/proactive.txt`, followed by a short
application tool contract. Observation and decision-making remain separate from
Executor. `observation_sources.py` provides read-only SQLite adapters:

- `email_inbox` observes received emails with both read/unread states. `read_email`
  returns all owned records or a complete thread, including drafts/sent messages,
  with boolean `read` flags. Inspection never marks email read. Stored thread/email
  IDs provide candidate task hints; the model can match another owned ACTIVE task
  by meaning, participants and chat context. Uncertain matches can ask the user.
- `reminders` observes persistent task schedules/statuses, including pending-action
  timing and overdue state (60-second grace). Because the schema has no task-type
  column, it includes other scheduled work such as email. It never sends reminders,
  reschedules them or takes recurrence away from existing controllers.

Full chat history and all task/action context are supplied during a changed cycle.
The `read_actions` tool retrieves owned actions and their structured context;
`read_user_context` rereads history/tasks. Preferences come from actual shared facts,
not an invented memory table. Reading a source uses SQLite read-only connections.
Observation receipts and effect IDs prevent replay across checks/restarts. A new
semantic task hint does not itself re-trigger an already processed email.

To enable local sources on an existing configuration and use a short demo interval:

```zsh
export AGENT_PROACTIVE_ENABLED=1
export AGENT_PROACTIVE_SOURCES=email_inbox,reminders
export AGENT_PROACTIVE_INTERVAL_SECONDS=30
python3 main.py
```

The default interval is 30 seconds. Empty sources or disabled proactive checks
make no model calls. Source data is passed to the configured LLM only when a changed
cycle needs inference. Proactive only stages `send_message` or `add_actions` for
existing tasks, then commits with `finish_check`; it cannot create new tasks, send
email or edit/cancel scheduled actions. Reminder delivery remains Executor's job.
Simulate an incoming message with `python3 email_store.py receive --from
colleague@example.test --content 'The document is ready.'` (optionally `--thread ID`).
