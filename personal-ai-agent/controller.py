"""Deterministic scheduled action controller. Run in the background or standalone."""

import logging
import os
from pathlib import Path
from threading import Event, Thread, Lock
from typing import Callable

from action_executor import ActionExecutor
from action_queue import ActionQueue, Claim, StaleClaim

logger = logging.getLogger(__name__)


class ActionController:
    def __init__(self, queue: ActionQueue, executor: ActionExecutor,
                 notify: Callable[[str], None], poll_seconds: float = 1,
                 max_workers: int = 2, *, deliver_to_terminal: bool = True,
                 execute_actions: bool = True) -> None:
        if poll_seconds <= 0 or type(max_workers) is not int or max_workers < 1:
            raise ValueError('Positive poll interval and worker count required')
        self.queue, self.executor, self.notify = queue, executor, notify
        self.deliver_to_terminal, self.execute_actions = deliver_to_terminal, execute_actions
        self.poll_seconds = poll_seconds
        self.stop_event = Event()
        self.thread: Thread | None = None
        self.max_workers = max_workers
        self.workers: list[Thread] = []
        self.delivery_lock = Lock()

    def deliver_notifications(self) -> None:
        if not self.deliver_to_terminal:
            return
        with self.delivery_lock:
            for notification in self.queue.pending_notifications():
                try:
                    self.notify(notification['message'])
                    self.queue.mark_delivered(notification['notification_id'])
                except Exception:
                    logger.exception('Notification delivery failed: %s', notification['notification_id'])
                    break

    def run_once(self) -> bool:
        self.deliver_notifications()
        if not self.execute_actions:
            return False
        claim = self.queue.claim_due()
        if claim is None:
            return False
        self.process_claim(claim)
        self.deliver_notifications()
        return True

    def process_claim(self, claim: Claim) -> None:
        logger.info('Action claimed: %s attempt=%s', claim.action.action_id, claim.action.attempt_count)
        try:
            self.executor.execute(claim)
        except StaleClaim:
            logger.info('Action no longer necessary: %s', claim.action.action_id)
        except Exception:
            logger.exception('Action failed; retry policy applies: %s', claim.action.action_id)
            self.queue.retry(claim)

    def dispatch_once(self) -> bool:
        """Poll independently of inference; claim only when a worker slot is free."""
        self.workers = [worker for worker in self.workers if worker.is_alive()]
        self.deliver_notifications()
        if not self.execute_actions:
            return False
        if len(self.workers) >= self.max_workers:
            logger.debug('controller_poll busy_workers=%s capacity=%s; remaining actions stay queued',
                         len(self.workers), self.max_workers)
            return False
        claim = self.queue.claim_due()
        if claim is None:
            return False
        worker = Thread(target=self.process_claim, args=(claim,),
                        name=f'action-{claim.action.action_id}', daemon=True)
        self.workers.append(worker)
        worker.start()
        return True

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                for _ in range(self.max_workers):
                    if self.stop_event.is_set() or not self.dispatch_once():
                        break
            except Exception:
                logger.exception('Controller iteration failed')
            self.stop_event.wait(self.poll_seconds)

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self.thread = Thread(target=self.run, name='action-controller', daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=1)


def main() -> None:
    from runtime_logging import configure_logging
    root = Path(__file__).resolve().parent
    configure_logging(root)
    queue = ActionQueue(Path(os.getenv('AGENT_DB_PATH', str(root / 'data' / 'agent.db'))).resolve(), os.getenv('AGENT_USER_ID', 'u1'))
    controller = ActionController(queue, ActionExecutor(queue, timezone=os.getenv('AGENT_TIMEZONE', 'Asia/Kolkata')),
                                  lambda message: None, deliver_to_terminal=False)
    try:
        controller.run()
    except KeyboardInterrupt:
        controller.stop()


if __name__ == '__main__':
    main()
