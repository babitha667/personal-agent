"""Validate proposed plans and persist tasks/actions in one transaction."""

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from database import connect
from models import Action, Task
from action_data import action_record
from scheduling import resolve_execution_time


def nonempty_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected non-empty text")
    return value.strip()


def validate_plan(proposals: object, user_id: str,
                  reference_time: datetime | None = None, *, initial_immediate: bool = False) -> tuple[list[Task], list[Action]]:
    if not isinstance(proposals, list) or len(proposals) > 10:
        raise ValueError("Expected at most ten tasks")
    now = datetime.now(timezone.utc)
    created_at = now.isoformat()
    tasks, actions = [], []
    for proposal in proposals:
        if not isinstance(proposal, dict) or set(proposal) != {"description", "period", "actions"}:
            raise ValueError("Invalid task fields")
        period = proposal["period"]
        if period is not None and (type(period) is not int or not 0 < period <= 2**63 - 1):
            raise ValueError("Period must be a positive integer or null")
        task = Task(
            task_id=str(uuid4()), user_id=user_id,
            description=nonempty_text(proposal["description"]),
            status="ACTIVE", period=period, created_at=created_at,
        )
        proposed_actions = proposal["actions"]
        if not isinstance(proposed_actions, list) or len(proposed_actions) != 1:
            raise ValueError("Each task must have exactly one initial action")
        for proposed_action in proposed_actions:
            if not isinstance(proposed_action, dict) or set(proposed_action) not in ({"exec_at", "description"}, {"exec_at", "description", "notification"}):
                raise ValueError("Invalid action fields")
            execution_time = resolve_execution_time(proposed_action['exec_at'], reference_time or now, now)
            notification = proposed_action.get('notification')
            if notification is not None:
                notification = nonempty_text(notification)
            actions.append(Action(
                action_id=str(uuid4()), task_id=task.task_id,
                exec_at=(now if initial_immediate else execution_time).astimezone(timezone.utc).isoformat(),
                description=nonempty_text(proposed_action["description"]),
                status="PENDING", attempt_count=0, created_at=created_at, context={"requested_time":execution_time.isoformat(), **({"legacy_message":notification} if notification else {})},
            ))
        tasks.append(task)
    return tasks, actions


def save_plan(db_path: Path, tasks: list[Task], actions: list[Action],
              cancel_task_ids: list[str] | None = None, user_id: str | None = None,
              queue_message_id: str | None = None, queue_token: str | None = None,
              require_new_description: str | None = None) -> None:
    with connect(db_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        if queue_message_id is not None:
            from message_queue import require_claim
            require_claim(connection, queue_message_id, queue_token)
        if require_new_description is not None and connection.execute(
            "SELECT 1 FROM tasks WHERE user_id=? AND status='ACTIVE' AND lower(trim(description))=lower(trim(?)) LIMIT 1",
            (user_id,require_new_description),
        ).fetchone():
            raise ValueError('An identical ACTIVE reminder already exists. Do not replay historical requests. '
                             'For relevant additional information use add_task_context; for changes use update_reminder '
                             'or reschedule_reminder. Only an explicitly requested extra copy may set allow_duplicate=true.')
        for task_id in cancel_task_ids or []:
            updated = connection.execute(
                "UPDATE tasks SET status = 'CANCELLED' "
                "WHERE task_id = ? AND user_id = ? AND status = 'ACTIVE'",
                (task_id, user_id),
            )
            if updated.rowcount != 1:
                raise ValueError("Task changed or is not owned by this user; retry with current context")
            connection.execute(
                "UPDATE actions SET status = 'CANCELLED' "
                "WHERE task_id = ? AND status IN ('PENDING', 'RUNNING')",
                (task_id,),
            )
        connection.executemany(
            """INSERT INTO tasks
            (task_id, user_id, description, status, period, created_at, last_executed_at)
            VALUES (:task_id, :user_id, :description, :status, :period,
                    :created_at, :last_executed_at)""",
            [asdict(task) for task in tasks],
        )
        connection.executemany(
            """INSERT INTO actions
            (action_id, task_id, exec_at, description, status, attempt_count, created_at, context)
            VALUES (:action_id, :task_id, :exec_at, :description, :status,
                    :attempt_count, :created_at, :context)""",
            [action_record(action) for action in actions],
        )
        if queue_message_id is not None:
            connection.execute("UPDATE message_inbox SET status='DONE',token=NULL,lease_until=NULL WHERE message_id=? AND token=?",
                               (queue_message_id, queue_token))
