"""Same-process live messaging, durable receipts and a terminal that never launches Meta."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chat_meta import main
from meta_chat import ChatMailbox
from persistent_meta import CodexAppServer, write_json


def test_queue_recovery_never_repeats_accepted_or_ambiguous_messages(tmp_path):
    mailbox=ChatMailbox(tmp_path/'chat')
    first=mailbox.enqueue('正常消息')
    accepted=mailbox.enqueue('已经接收')
    mailbox.save({**accepted,'status':'accepted'})
    sending=mailbox.enqueue('中断时正在投递')
    mailbox.save({**sending,'status':'sending'})
    mailbox.recover()
    assert [x['id'] for x in mailbox.pending('new-turn')]==[first['id']]
    assert mailbox.records()[-1]['status']=='delivery_unknown'


def test_chat_does_not_start_another_process_and_retries_only_after_stale_turn(tmp_path):
    executable=tmp_path/'fake_codex'
    executable.write_text('''#!/usr/bin/env python3
import json,sys
def send(x): print(json.dumps(x),flush=True)
turn=0
for line in sys.stdin:
    m=json.loads(line); method=m.get('method')
    if method=='initialize':send({'id':m['id'],'result':{}})
    elif method=='turn/start':
        turn+=1
        send({'id':m['id'],'result':{'turn':{'id':str(turn)}}})
        send({'method':'turn/started','params':{'threadId':'same-thread','turn':{'id':str(turn)}}})
    elif method=='turn/steer':
        assert m['params']['threadId']=='same-thread'
        assert m['params']['expectedTurnId']==str(turn)
        assert m['params']['input'][0]['text']=='hello same Meta'
        if turn==1:
            send({'id':m['id'],'error':{'message':'active turn id mismatch'}})
        else:
            send({'id':m['id'],'result':{'turnId':str(turn)}})
        send({'method':'turn/completed','params':{'threadId':'same-thread','turn':{'id':str(turn),'status':'completed'}}})
''')
    executable.chmod(0o755)
    mailbox=ChatMailbox(tmp_path/'chat')
    record=mailbox.enqueue('hello same Meta')
    server=CodexAppServer(codex_home=tmp_path,workspace=tmp_path,journal=tmp_path/'rpc',
                          executable=str(executable),chat=mailbox)
    try:
        server.start();pid=server.process.pid
        server.turn('same-thread','first')
        assert mailbox.records()[0]['status']=='queued'
        assert mailbox.pending('1')==[]
        server.turn('same-thread','next')
        result=mailbox.records()[0]
        assert result['status']=='accepted' and result['id']==record['id']
        assert result['thread_id']=='same-thread' and result['turn_id']=='2'
        assert server.process.pid==pid and server.process.poll() is None
        assert mailbox.pending('3')==[]
    finally:
        server.close()


def test_delivery_error_is_a_message_receipt_not_an_experiment_exception(tmp_path):
    mailbox=ChatMailbox(tmp_path/'chat')
    record=mailbox.enqueue('hello')
    server=CodexAppServer(codex_home=tmp_path,workspace=tmp_path,journal=tmp_path/'rpc',chat=mailbox)
    server.chat_requests['request-1']=record
    server.inbox.put({'id':'request-1','error':{'message':'example error'}})
    server.receive()
    assert mailbox.records()[0]['status']=='delivery_error'


def test_send_cli_only_queues_message(tmp_path,capsys):
    run=tmp_path/'run'
    write_json(run/'meta_session/state.json',{'thread_id':'existing','interactive_chat':True})
    assert main(['--run-dir',str(run),'--send','请汇报当前进展'])==0
    result=json.loads(capsys.readouterr().out)
    assert result['status']=='queued'
    assert result['text']=='请汇报当前进展'
    assert list((run/'meta_session/chat').glob('*.json'))
    assert not (run/'meta_session/app_server').exists()


def test_quit_interface_does_not_change_experiment(tmp_path,monkeypatch,capsys):
    run=tmp_path/'run'
    path=run/'meta_session/state.json'
    write_json(path,{'thread_id':'existing','status':'running','interactive_chat':True})
    original=path.read_bytes()
    monkeypatch.setattr('builtins.input',lambda _: '/quit')
    assert main(['--run-dir',str(run),'--history','0'])==0
    assert path.read_bytes()==original
    assert not (run/'meta_session/chat').exists()
    assert '实验继续运行' in capsys.readouterr().out
