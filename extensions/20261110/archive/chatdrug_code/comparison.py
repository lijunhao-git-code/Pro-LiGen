"""Matched task adapter; native first-parsed ChatDrug and Pro-LiGen selection remain distinct."""
import argparse,copy,fcntl,hashlib,json,os,re,sqlite3,threading,time,sys
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from common import ROOT,read,read_jsonl,sha,dump,now,PROPERTIES,evaluate_smiles,candidate_records,choose,feedback
from durable_api import Runner,Pause,TechnicalFailure

# Matches the official small-molecule parser's alphabet and minimum length.
PATTERN=re.compile(r'[0-9BCOHNSOPrIFlanocs@+\.\-\[\]\(\)\\/%=#$]{6,}')
def occurrences(text):return PATTERN.findall(text or '')
def native_first(text,seed,last_reference=None):
 return next((s for s in occurrences(text) if s!=seed and s!=last_reference),None)

class Retrieval:
 def __init__(self):self.pools=read(ROOT/'inputs/pools75.json');self.cache={};self.fingerprints={};self.lock=threading.RLock()
 def get(self,key):
  with self.lock:
   if key not in self.cache:
    seq=self.pools[key]
    for s in seq:
     if s not in self.fingerprints:self.fingerprints[s]=AllChem.GetMorganFingerprint(Chem.MolFromSmiles(s),2)
    fps=[self.fingerprints[s] for s in seq]
    self.cache[key]=(seq,fps)
   return self.cache[key]
 def closest(self,key,failed_smiles):
  seq,fps=self.get(key);q=AllChem.GetMorganFingerprint(Chem.MolFromSmiles(failed_smiles),2)
  scores=DataStructs.BulkTanimotoSimilarity(q,fps);i=int(np.argmax(scores))
  return seq[i],float(scores[i])

def instruction(entry):
 prop=entry['property'];label=PROPERTIES[prop]['label']
 if prop=='ring':goal=f'exactly {int(entry["target_center"])} graph rings, one fewer than the original seed'
 else:goal=f'an RDKit {label} value in [{entry["target_lower"]:.8f}, {entry["target_upper"]:.8f}] including endpoints'
 carbon='' if prop=='ring' else ' and contain carbon'
 return (f'Edit the original seed molecule {entry["seed_smiles"]} to have {goal}. '
  'The output molecule should be similar to the original seed. '
  f'It must be a nonempty single connected molecule{carbon}. Do not copy the original seed or supplied reference molecules. '
  'Do not claim experimental measurements or docking results.')
def initial_prompt(entry):return instruction(entry)+' Give me five molecules in SMILES only and list them using bullet points. No explanation is needed.'
def revision_prompt(entry,failed,ref):
 return (f'Your provided sequence {failed} is not correct. We find a sequence {ref} which is correct and similar to the molecule you provided. '
  'Can you give me a new molecule? '+instruction(entry)+' Return one SMILES only. No explanation is needed.')

class ComparisonRunner(Runner):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.contexts=read(ROOT/'inputs/contexts75.json');self.retrieval=Retrieval()
 def complete(self,call_id,body,started,response_path):
  super().complete(call_id,body,started,response_path)
  fp=body.get('system_fingerprint')
  if fp:
   old=self.db.execute("SELECT value FROM meta WHERE key='system_fingerprint'").fetchone()
   if old and old[0]!=fp:self.halt('provider_fingerprint_changed')
   if not old:self.db.execute('INSERT INTO meta VALUES (?,?)',('system_fingerprint',fp));self.db.commit()
 def reply(self,slot,attempt,turn,messages):
  b=self.call(slot,attempt,turn,messages);ch=b['choices'][0];text=ch['message'].get('content') or ''
  if ch.get('finish_reason')!='stop' or not text.strip():raise TechnicalFailure('empty_or_truncated_response')
  return text
 def execute(self,slot,entry,attempt,state):
  messages=copy.deepcopy(self.contexts[entry['prefix_key']]);rows=[];refs=list(entry['exemplars']);trace=[]
  if slot['group']=='G':
   for turn in [4,5]:
    prompt=entry['prompt'][turn-1]
    if turn==5:prompt+=feedback(rows,entry)
    messages.append(dict(role='user',content=prompt));text=self.reply(slot['slot_id'],attempt,turn,messages)
    messages.append(dict(role='assistant',content=text));proposed,extra=candidate_records(text,entry,turn);rows.extend(proposed)
    trace.append(dict(turn=turn,provided_references=refs,parsed_candidates=len(proposed),ignored_extra_tags=extra))
    state.update(current_attempt=attempt,completed_turn=turn,candidates=rows,status='running',last_update_utc=now());dump(self.root/'logs/slot_states'/f'{slot["slot_id"]}.json',state)
   return choose(rows),rows,refs,trace,'eligible_output' if choose(rows) else 'no_eligible_candidate'
  messages.append(dict(role='user',content=initial_prompt(entry)));last_ref=None
  for index in range(3):
   turn=4+index;text=self.reply(slot['slot_id'],attempt,turn,messages);messages.append(dict(role='assistant',content=text))
   extracted=occurrences(text);first=native_first(text,entry['seed_smiles'],last_ref)
   dynamic=dict(entry,exemplars=list(refs));stage=[]
   for pos,s in enumerate(extracted,1):stage.append(evaluate_smiles(s,dynamic,turn,pos))
   rows.extend(stage)
   selected_index=next((i for i,s in enumerate(extracted) if s==first),None) if first else None
   native=stage[selected_index] if selected_index is not None else None
   for r in stage:r['native_evaluated']=(r is native)
   event=dict(turn=turn,extracted_count=len(extracted),native_candidate_position=selected_index+1 if selected_index is not None else None,
    provided_references=list(refs),native_reason=native.get('reason') if native else 'unparsed',C=2)
   trace.append(event)
   state.update(current_attempt=attempt,completed_turn=turn,candidates=rows,trace=trace,status='running',last_update_utc=now());dump(self.root/'logs/slot_states'/f'{slot["slot_id"]}.json',state)
   if native is None:return None,rows,refs,trace,'native_unparsed'
   if not native['valid']:return None,rows,refs,trace,'native_invalid'
   if native['eligible']:return native,rows,refs,trace,'eligible_output'
   if index==2:return None,rows,refs,trace,'native_round_limit'
   ref,similarity=self.retrieval.closest(entry['prefix_key'],native['canonical'])
   check=evaluate_smiles(ref,dict(entry,exemplars=[]),turn,1)
   assert check['valid'] and check['property_hit'] and not check['copy_seed']
   refs.append(ref);last_ref=ref;event.update(retrieved_reference=ref,retrieval_tanimoto=similarity)
   messages.append(dict(role='user',content=revision_prompt(entry,first,ref)))
  raise AssertionError('unreachable')
 def slot(self,slot,entry):
  path=self.root/'results'/f'{slot["slot_id"]}.json'
  if path.exists():return read(path)
  state_path=self.root/'logs/slot_states'/f'{slot["slot_id"]}.json'
  state=read(state_path) if state_path.exists() else dict(**slot,next_attempt=1,attempts=[],started_utc=now())
  for attempt in range(state['next_attempt'],3):
   try:
    selected,rows,refs,trace,status=self.execute(slot,entry,attempt,state)
    r=dict(**slot,started_utc=state['started_utc'],finished_utc=now(),technical_complete=True,status=status,
     ok=selected is not None,canonical=selected['canonical'] if selected else None,selected_candidate=selected,candidates=rows,
     provided_references=refs,trace=trace,successful_attempt=attempt,earlier_attempts=state['attempts'],
     common_context_sha256=entry['prefix_sha256'])
    dump(path,r);state.update(status='complete',last_update_utc=now());dump(state_path,state);return r
   except Pause as ex:
    state.update(status='paused',pause_reason=str(ex),current_attempt=attempt,next_attempt=attempt,last_update_utc=now());dump(state_path,state)
    return dict(slot_id=slot['slot_id'],status='paused',reason=str(ex))
   except TechnicalFailure as ex:
    state['attempts'].append(dict(attempt=attempt,status='technical_failure',reason=str(ex)))
    state.update(next_attempt=attempt+1,status='retry_pending',last_update_utc=now());dump(state_path,state)
   except Exception as ex:
    with self.lock:self.reason='unexpected_'+type(ex).__name__;self.stop.set()
    state.update(status='paused_unexpected',error_class=type(ex).__name__,next_attempt=attempt,last_update_utc=now());dump(state_path,state)
    return dict(slot_id=slot['slot_id'],status='paused',reason=self.reason)
  result=dict(**slot,status='failed_after_bounded_attempts',ok=False,technical_complete=False,canonical=None,
   selected_candidate=None,candidates=[],earlier_attempts=state['attempts'],finished_utc=now(),provided_references=[],trace=[])
  dump(path,result);return result
 def summary(self,mode):
  results=[read(p) for p in (self.root/'results').glob('*.json')]
  with self.lock:
   states=dict(self.db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state').fetchall())
   total=float(self.db.execute('SELECT COALESCE(SUM(charge_estimate),0) FROM calls').fetchone()[0])
  groups={}
  for prop in ['ring','logp','tpsa']:
   for g in ['G','CD']:
    subset=[r for r in results if r['property']==prop and r['group']==g]
    groups[f'{prop}_{g}']=dict(completed=len(subset),delivered=sum(r['ok'] for r in subset))
  value=dict(time_utc=now(),mode=mode,planned=12 if mode=='pilot' else 750,completed=len(results),delivered=sum(r['ok'] for r in results),
   technical_complete=sum(r.get('technical_complete',False) for r in results),stop_reason=self.reason,call_states=states,groups=groups,
   archived_ratecard_cost_estimate_cny=total,all_paid_responses_durable=True)
  dump(self.root/'logs/generation_status.json',value);return value

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--full',action='store_true');ap.add_argument('--workers',type=int,default=None);args=ap.parse_args()
 assert args.pilot!=args.full
 RDLogger.DisableLog('rdApp.*')
 lock=read(ROOT/'inputs/generation_locked.json');assert all(sha(ROOT/n)==v for n,v in lock['files'].items())
 if args.full:assert read(ROOT/'analysis/pilot_review.json')['allow_full_run'] is True
 mode='pilot' if args.pilot else 'full';root=ROOT/'pilot' if args.pilot else ROOT
 (root/'logs').mkdir(parents=True,exist_ok=True)
 guard=(root/'logs/generation.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
 previous=root/'logs/generation_status.json'
 if previous.exists():assert read(previous).get('stop_reason') in [None,'insufficient_balance','API_insufficient_balance_402','balance_check_failed'],'Reconcile prior halt before resume'
 entries={e['slot_id']:e for e in read_jsonl(ROOT/'inputs/entries750.jsonl')};schedule=read(ROOT/'inputs/schedule750.json')
 selected=[s for s in schedule if s['pilot']] if args.pilot else schedule
 pending=[s for s in selected if not (root/'results'/f'{s["slot_id"]}.json').exists()]
 runner=ComparisonRunner(root=root);workers=args.workers or (4 if args.pilot else 24)
 print(json.dumps(dict(status='starting',mode=mode,pending=len(pending),workers=workers,time_utc=now())),flush=True)
 queue=iter(pending)
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures={}
  def enqueue():
   if runner.stop.is_set():return
   try:slot=next(queue)
   except StopIteration:return
   futures[pool.submit(runner.slot,slot,entries[slot['slot_id']])]=slot
  for _ in range(workers):enqueue()
  while futures:
   done,_=wait(futures,timeout=20,return_when=FIRST_COMPLETED)
   if not done:print('HEARTBEAT '+json.dumps(runner.summary(mode)),flush=True);continue
   for f in done:
    slot=futures.pop(f)
    try:r=f.result();print('SLOT '+slot['slot_id']+' '+r['status'],flush=True)
    except Exception as ex:runner.stop.set();runner.reason='worker_'+type(ex).__name__
    runner.summary(mode);enqueue()
 value=runner.summary(mode);print('FINISHED '+json.dumps(value),flush=True)
 if runner.stop.is_set():raise SystemExit(2)
if __name__=='__main__':main()
