"""Terminal input stays responsive while one worker plans queued messages."""
import os
import json
from pathlib import Path
from threading import Lock

from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue
from controller import ActionController
from database import connect, initialize_task_tables
from message_queue import MessageQueue, MessageWorker
from models import Message
from storage import MessageStore
from runtime_logging import configure_logging


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    log_path = configure_logging(project_dir)
    db_path = Path(os.getenv('AGENT_DB_PATH', str(project_dir / 'data' / 'agent.db'))).resolve()
    initialize_task_tables(db_path)
    contextualizer = Contextualizer(MessageStore(db_path), user_id=os.getenv('AGENT_USER_ID', 'u1'),
                              timezone=os.getenv('AGENT_TIMEZONE', 'Asia/Kolkata'))
    output_lock = Lock()

    def notify(text):
        with output_lock:
            print(f'\nAgent: {text}', flush=True)

    inbox = MessageQueue(db_path, contextualizer.user_id, lease_seconds=contextualizer.planning_timeout + 60)
    worker = MessageWorker(inbox, contextualizer)
    print('Personal AI Agent')
    print(f'Debug log: {log_path}')
    print(f'Model: {contextualizer.llm.model} ({contextualizer.llm.tool_mode} tools)')
    print('Listening for messages. Press Enter to send, or Ctrl+C to stop.')
    print('Messages are saved to the inbox and processed in order.')
    execute_actions = os.getenv('AGENT_WORKER_ENABLED', '1') != '0'
    queue = ActionQueue(db_path, contextualizer.user_id)
    controller = ActionController(queue, ActionExecutor(queue, timezone=contextualizer.timezone.key), notify,
                                  execute_actions=execute_actions)
    controller.start()
    print('Action controller running (polls due actions every second).' if execute_actions
          else 'Notification listener running; actions execute in the standalone controller.')
    from continuous_user_processor import from_environment
    continuous = from_environment(db_path, contextualizer.user_id)
    continuous.start()
    with connect(db_path,read_only=True) as db:
        monitoring=db.execute('SELECT proactive_enabled,sources,interval_seconds FROM continuous_users WHERE user_id=?',(contextualizer.user_id,)).fetchone()
    if monitoring and monitoring['proactive_enabled'] and json.loads(monitoring['sources']):
        print(f"Proactive monitoring: {', '.join(json.loads(monitoring['sources']))} (checks every {monitoring['interval_seconds']:g}s).")
    else:
        print('Proactive monitoring is disabled or has no granted sources.')
    worker.start()
    try:
        while True:
            try:
                text = input('You: ').strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text:
                message = Message.from_content(text)
                with inbox.submission_lock:
                    inbox.enqueue(message)
                    controller.deliver_notifications()
    finally:
        continuous.stop()
        worker.stop()
        if controller:
            controller.stop()
        with connect(db_path) as db:
            remaining = db.execute("SELECT COUNT(*) FROM message_inbox WHERE user_id=? AND status IN ('PENDING','RUNNING')",
                                   (contextualizer.user_id,)).fetchone()[0]
        print(f'\nGoodbye! {remaining} unfinished message(s) saved for the next run.' if remaining else '\nGoodbye!')


if __name__ == '__main__':
    main()
