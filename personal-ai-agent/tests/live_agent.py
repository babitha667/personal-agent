"""Opt-in real-model smoke check; isolated DB, API calls billed to configured key."""
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from contextualizer import Contextualizer
from action_executor import ActionExecutor
from action_queue import ActionQueue
from database import connect
from models import Message
from storage import MessageStore
from task_store import TaskStore


def main():
    with tempfile.TemporaryDirectory() as folder:
        path=Path(folder)/'agent.db'
        contextualizer=Contextualizer(MessageStore(path))
        queue=ActionQueue(path)
        executor=ActionExecutor(queue)
        print('Testing configured model:',contextualizer.llm.model)
        answer=contextualizer.receive_message(Message.from_content('Hey there!'))
        assert answer.reply or answer.actions, 'No chat response/action'
        if answer.actions:
            executor.execute(queue.claim_due())
        assert not TaskStore(path).user_context('u1'), 'Greeting created persistent task'
        request=Message.from_content('Remind me to drink water in five minutes')
        result=contextualizer.receive_message(request)
        assert len(result.tasks)==1 and len(result.actions)==1, 'No task/kickoff pair'
        assert result.actions[0].context['request']['message']==request.message, 'Original request missing'
        executor.execute(queue.claim_due())
        with connect(path) as db:
            pending=[dict(row) for row in db.execute("SELECT * FROM actions WHERE status='PENDING'")]
        assert len(pending)==1 and pending[0]['description'], 'Executor did not schedule a reminder action'
        before=pending[0]
        contextualizer.receive_message(Message.from_content('I have a prawns allergy'))
        executor.execute(queue.claim_due())
        with connect(path) as db:
            after=dict(db.execute('SELECT * FROM actions WHERE action_id=?',(before['action_id'],)).fetchone())
        assert after==before, 'Unrelated personal fact changed generic water reminder'
        assert len(TaskStore(path).user_context('u1'))==1, 'Personal fact created unrelated task'
        contextualizer.receive_message(Message.from_content('Cancel the water reminder'))
        executor.execute(queue.claim_due())
        assert TaskStore(path).user_context('u1')[0]['status']=='CANCELLED', 'Task not cancelled'
        print('PASS: chat, task/kickoff, executor scheduling, unrelated information, cancellation.')
        print('No production records were used. Five-minute delivery was scheduled, not waited for.')

if __name__=='__main__':
    main()
