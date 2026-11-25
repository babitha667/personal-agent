"""Structured incoming messages shared by the terminal and Contextualizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4
from typing import Literal


@dataclass(frozen=True)
class Message:
    id: str
    timestamp: str
    message: str

    @classmethod
    def from_content(cls, message: str) -> "Message":
        if not message.strip():
            raise ValueError("Message content must not be blank.")
        return cls(
            id=str(uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            message=message,
        )


@dataclass(frozen=True)
class ActionContext:
    incoming_message: Message
    missing_information: list[str]
    tasks: list[Task] = field(default_factory=list)
    actions: list[Action] = field(default_factory=list)
    cancelled_task_ids: list[str] = field(default_factory=list)
    reply: str | None = None
    pending_activity: str | None = None


@dataclass(frozen=True)
class Task:
    task_id: str
    user_id: str
    description: str
    status: Literal["ACTIVE", "COMPLETED", "FAILED", "CANCELLED"]
    period: int | None
    created_at: str
    last_executed_at: str | None = None


@dataclass(frozen=True)
class Action:
    action_id: str
    task_id: str | None
    exec_at: str
    description: str
    status: Literal["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]
    attempt_count: int
    created_at: str
    context: dict = field(default_factory=dict)
