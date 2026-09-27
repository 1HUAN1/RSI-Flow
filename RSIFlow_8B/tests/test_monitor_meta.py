"""The monitor must display facts without touching a running experiment."""
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from monitor_meta import EventTail, EventView, main, receipt_summary


def event(method, item=None, **params):
    return {'method':method,'params':dict(params, **({'item':item} if item is not None else {}))}


def test_streaming_message_is_not_duplicated_at_completion():
    output=io.StringIO();view=EventView(output)
    view.handle(event('item/agentMessage/delta',itemId='m',delta='正在检查'))
    view.handle(event('item/agentMessage/delta',itemId='m',delta='候选。'))
    view.handle(event('item/completed',{'type':'agentMessage','id':'m','text':'正在检查候选。'}))
    assert output.getvalue().count('正在检查候选。')==1


def test_monitor_does_not_show_system_prompts_or_raw_reasoning():
    output=io.StringIO();view=EventView(output)
    view.handle(event('item/started',{'type':'userMessage','text':'PRIVATE_INPUT'}))
    view.handle(event('item/reasoning/textDelta',delta='RAW_REASONING'))
    assert output.getvalue()==''


def test_monitor_shows_native_shell_and_file_edits_in_history(tmp_path):
    output=io.StringIO();view=EventView(output)
    records=[event('item/completed',{'type':'commandExecution','command':'python check.py',
        'status':'completed','exitCode':0,'aggregatedOutput':'NATIVE_CODING_OK'}),
        event('item/completed',{'type':'fileChange','status':'completed',
                               'changes':[{'path':'check.py','kind':{'type':'add'}}]})]
    path=tmp_path/'events.jsonl'
    path.write_text(''.join(json.dumps(record)+'\n' for record in records))
    for record in EventTail(path).history(5):
        view.handle(record,replay=True)
    assert 'python check.py' in output.getvalue()
    assert 'NATIVE_CODING_OK' in output.getvalue()
    assert '原生文件修改' in output.getvalue()


def test_tool_receipt_status_overrides_successful_transport():
    item={'success':True,'status':'completed','contentItems':[{'type':'inputText','text':
          'Tool receipt at /run/receipt.json:\n'+json.dumps({'status':'tool_error','error':'missing file'})+'\nReminder'}]}
    result=receipt_summary(item)
    assert 'status=tool_error' in result and 'missing file' in result
    assert '/run/receipt.json' in result


def test_partial_utf8_line_and_rotation_are_followed(tmp_path):
    path=tmp_path/'events.jsonl';tail=EventTail(path)
    assert tail.history(5)==[]
    raw=json.dumps(event('item/agentMessage/delta',itemId='m',delta='你好'),ensure_ascii=False).encode()+b'\n'
    cut=raw.index('你'.encode())+1
    path.write_bytes(raw[:cut])
    assert tail.poll()==[]
    with path.open('ab') as stream:stream.write(raw[cut:])
    assert tail.poll()[0]['params']['delta']=='你好'
    assert tail.poll()==[]
    path.rename(tmp_path/'old.jsonl')
    path.write_bytes(raw)
    assert len(tail.poll())==1


def test_history_is_bounded_and_new_events_are_not_lost(tmp_path):
    path=tmp_path/'events.jsonl'
    def row(i):return json.dumps(event('item/completed',{'type':'agentMessage','id':str(i),'text':str(i)}))+'\n'
    path.write_text(''.join(row(i) for i in range(5)))
    tail=EventTail(path)
    assert [x['params']['item']['text'] for x in tail.history(2)]==['3','4']
    with path.open('a') as stream:stream.write(row(5))
    assert tail.poll()[0]['params']['item']['text']=='5'


def test_once_creates_no_files_and_starts_no_process(tmp_path,capsys):
    run=tmp_path/'not_started'
    assert main(['--run-dir',str(run),'--once'])==0
    assert not run.exists()
    output=capsys.readouterr().out
    assert '不在运行' in output and '只读' in output
