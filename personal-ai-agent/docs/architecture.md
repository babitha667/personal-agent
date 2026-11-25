# Agent architecture

The default transport is DeepSeek Chat Completions (`deepseek-flash`) with native
function tools. `LLM_PROVIDER=ollama` selects local inference. Cloud inference
receives the supplied conversation/task context; SQLite remains local.

## Responsibilities

Contextualizer collects relevant user context and classifies each incoming message.
Its only tool is create_action: exactly one immediate handoff. For a new request,
new_task in that call creates the initial task record in the same transaction.
Existing tasks/actions are never edited by Contextualizer. It does not communicate directly.
Generic conversation uses task_id=null and still goes through Executor.

Action Executor owns subsequent task and action management, operations, progress,
follow-ups and user communication. Its native tools include read_tasks/read_actions,
select_task/create_task, update_task/cancel_task, update_action/cancel_action,
add_actions/send_message, local email tools and finish_action. Taskless ordinary
chat cannot create tasks. Selecting an existing task is scoped to the same user.

Proactive is separate. It observes permitted sources, sends useful suggestions via
the existing outbox, or enqueues an immediate action for an existing ACTIVE task.
It cannot create tasks, mutate external applications or execute operations. Approval
of suggestions comes back through the normal incoming message flow.

Deterministic controllers retain dispatch, recurrence/proactive intervals, leases,
retry scheduling and outgoing-message delivery. No new queues or frameworks are added.

## Atomic management and execution

planner_operations.py commits initial task creation, one handoff, source linkage,
message receipt and inbox completion together. Repeated input cannot create duplicates.
The original request and receipt are preserved in context, without timing calculation.

executor_management.py stages task/action mutations in an owned working view.
finish_action checks the current lease and compares original rows under BEGIN IMMEDIATE
before committing management changes, local email effects, follow-ups, results and
outbox messages. A stale task/action snapshot aborts all effects for a fresh retry.
Replacing an action cancels it and removes its lease, fencing an older worker.
Context updates preserve timing unless Executor supplies a new time. There is no
literal phrase match: semantic relevance belongs to the LLM; ownership and concurrency
validation belong to deterministic code.

The bounded Executor uses six working turns plus a completion-only turn, at most
sixteen tools. An invalid batch rolls back its staging; no staged success message
is delivered. read_actions exposes structured context for the associated task.
The action input carries relevant history rather than the full global conversation.
All past task actions and committed outcomes propagate as a flat past_actions list.

## Scheduling and task state

Kickoff actions run NOW, acknowledge future requests and schedule one next follow-up.
Follow-ups use description, structured context and exec_at; every due action invokes
Executor. Relative kickoff delays anchor to receipt; continuation delays anchor to now.
Task changes update/replace pending work rather than duplicating tasks. Cancellation
invalidates operational actions; already queued user handoffs can still be evaluated.

Action completion does not implicitly complete a task in the native Executor path.
An explicit COMPLETED task status requires a verified goal and no pending operational
work. WAITING keeps the task ACTIVE. Recurring tasks remain ACTIVE after occurrences;
Continuous User Processor alone produces the next occurrence via the durable inbox.
Administrative updates do not advance last_executed_at, and an exhausted administrative
handoff cannot cancel an otherwise valid scheduled task.

Inbox leases, bounded transport retries, ownership checks, transactional operation
batches, and commit receipts protect against partial writes and duplicate task
creation. Model intent can still be wrong. Terminal delivery is at least once:
a crash after printing but before marking delivered may repeat an outgoing message.
Only one chat frontend per user is supported.

Existing databases migrate automatically to nullable task links plus taskless
owner metadata. Action IDs, dependent leases, and existing records are retained.

## Verification and limits

`python3 -m unittest discover -s tests -q` uses disposable databases and controlled
model responses. Tests include current generic planning/execution and frozen
legacy regression fixtures, which are not the current runtime planner.
`python3 tests/live_agent.py` checks the current configured model with a temporary
database. It makes API calls when DeepSeek is selected and requires the key in
that terminal. Backend tests alone do not establish live model semantic quality.

Full history/context grows without compaction. External application integrations
are deferred; the executor currently handles notifications and follow-ups.

## Continuous observation

`continuous_user_processor.py` registers/discovers users and performs deterministic
recurrence checks. `continuous_users` persists per-user enablement, source grants,
intervals, next check time, and leased tokens. `recurring_triggers` atomically links
one task execution basis to its incoming message. Existing outstanding actions and
pending incoming triggers suppress duplicate occurrences.

`proactive.py` polls only permitted read-only sources and fingerprints snapshots.
`proactive_seen` records processed versions; `proactive_effects` records per-version,
per-user message/action effects. Unchanged data bypasses inference. A bounded
native-tool loop stages suggestions or immediate actions on existing active tasks,
then commits them and the observation receipts together. Permissions and leases
are rechecked before reading, inference, and commit. No task creation or external
mutation functions are exposed. Supplied external content is untrusted data.

The continuous/proactive flow is enabled by default, with local read-only mailbox
and reminder sources granted to newly registered users; saved source revocations persist. Empty or unchanged snapshots cause no inference. Application
integrations are deferred; tests inject read-only fixtures. Future source grants
remain explicit and per-user, independently of the controller enablement flag.
The source protocol allows later adapters without replacing queues or agents.

## Action contract and history

Action records expose `description` and structured `context`; no instructions field
remains. The executor owns procedure. Context is encoded only at the SQLite
boundary, parsed for native tool/model payloads, and validated as an object on input.
The schema migration renames the old column and preserves its contents and IDs.

`action_history.py` builds flat task history from committed records. Every next
action carries all prior actions plus the current committed result in
`context.past_actions`; history entries exclude their own ancestor lists. Outcome,
reason, actual outgoing message and created follow-up IDs are recorded together
with queue completion. Existing nested facts merge with newly supplied context.
The planner and proactive agent also attach existing task history when creating
continuation actions. No model-invented action history is accepted.

Kickoff context has an explicit phase. Required requested-time metadata distinguishes
immediate preparation from deferred execution, and Python validates future kickoff
completion against its queued delivery. Backend tests include rejecting the observed
incorrect early-send/complete sequence for two- and five-minute requests. Semantic
extraction of requested timing remains the model's responsibility; this is not a
keyword parser or proof that every natural-language request is understood correctly.

## Local mailbox increment

`email_store.py` defines the append-only message/thread schema and a CLI to inspect
records or inject simulated incoming messages. Ownership is keyed by `user_id`.
Native executor tools `draft_email`, `send_email`, and `read_email` extend the generic tools;
Contextualizer creates only initial task records and immediate handoffs. Sender identity is configured
by the backend with `AGENT_EMAIL_ADDRESS`, otherwise a stable local demo address.

Outgoing mail and explicit read-state changes are staged by ExecutorTools and
committed within ActionQueue.finish with the fenced action, outgoing chat, task
state, and verified result history. One outgoing message per action and a unique
sending-action link prevent retry duplicates. Read-only inspection preserves unread
state. Incoming content has no authority to select users, tools, or permissions.
The local proactive mailbox source is installed. No external email transport
or automatic reply workflow is installed; those can be added independently.

Draft lifecycle is explicit: DRAFT is unsent, SENT is a committed local outgoing
message, and RECEIVED is simulated inbound mail. Subject is a persisted field.
Editing or sending an existing draft preserves identity and thread; optimistic
revision checks prevent stale edits and repeat sends. SENT/RECEIVED content is
immutable through the email tools. Existing committed outgoing rows migrate as
SENT; no historical sends are recast as drafts. Historical missing subjects are
not fabricated. Task last_result exposes verified effects, and recent dialogue is
highlighted in planner input to anchor short follow-up answers.

Each due action invokes the executor. There is no prepared-notification bypass.
Action fields use description, structured context, and exec_at; obsolete scheduled_for
and notification columns migrate into context and are removed.


Local observation adapters are in `observation_sources.py`. Received emails retain
read/unread flags; reading never consumes them. Stable email versions ignore derived
task-link hints to prevent self-trigger loops. Thread/email ID hints augment semantic
matching; they are not an exact-ID restriction on selecting an owned ACTIVE task.
Reminder observations describe persistent schedules, with dispatcher/recurrence
ownership explicit. They do not include newly staged proactive handoffs as reminders.

On changed cycles, Proactive receives full user chat history and task/action context.
Read-only native tools expose owned email threads, actions and user context. The native
effects are send_message and add_actions; finish_check commits them atomically. An
invalid tool batch restores staging. The registered local sources are permission-scoped
and use SQLite mode=ro; no real email/calendar account or automatic sending is added.
