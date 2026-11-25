"""Contextualizer: context collection and exactly one immediate executor handoff."""
import os
from copy import deepcopy
from zoneinfo import ZoneInfo

from database import initialize_task_tables
from llm import ChatClient, create_client
from models import ActionContext, Message
from outbox import initialize_outbox
from planner_operations import PLANNER_TOOLS, initialize_planner
from planning import nonempty_text
from storage import MessageStore


class Contextualizer:
    def __init__(self, message_store: MessageStore, llm: ChatClient | None = None,
                 user_id: str = 'u1', timezone: str = 'Asia/Kolkata') -> None:
        self.message_store = message_store
        self.llm = llm if llm is not None else create_client()
        self.user_id = nonempty_text(user_id)
        self.timezone = ZoneInfo(timezone)
        self.tool_definitions = deepcopy(PLANNER_TOOLS)
        self.planning_timeout = float(os.getenv('AGENT_PLANNING_TIMEOUT_SECONDS','120'))
        if not 0 < self.planning_timeout <= 300:
            raise ValueError('AGENT_PLANNING_TIMEOUT_SECONDS must be between 0 and 300')
        initialize_task_tables(message_store.db_path)
        initialize_outbox(message_store.db_path)
        initialize_planner(message_store.db_path)

    def receive_message(self, message: Message, *, queue_token: str | None = None) -> ActionContext:
        from generic_contextualizer import receive
        return receive(self,message,queue_token)
