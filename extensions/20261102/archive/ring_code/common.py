"""Atomic IO and unchanged original Study-B chemistry adapters."""
from pathlib import Path
import json,os
from original_chemistry import PROJECT,EXP,sha,now,read_jsonl,canon,properties,scorer
from original_chemistry import candidate_records as original_candidates
from original_chemistry import select_candidate,optimization_feedback
ROOT=Path(__file__).resolve().parents[1]
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
def candidate_records(content,entry,turn):return original_candidates(content,turn,entry)
def choose(rows,policy='best_QED'):
    items=sorted(rows,key=lambda r:(r['turn'],r['candidate_position']))
    if policy=='first_eligible':return next((r for r in items if r['eligible']),None)
    return select_candidate(items)
def feedback(rows,entry):return optimization_feedback(rows,entry)
