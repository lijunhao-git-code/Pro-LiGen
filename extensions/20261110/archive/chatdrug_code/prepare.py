"""Prepare a frozen, paired three-condition comparison without paid generation."""
from pathlib import Path
import hashlib,json,random,shutil,sys
import pandas as pd
from rdkit import Chem,RDLogger,rdBase
from common import ROOT,read,read_jsonl,dump,sha,now,properties,PROPERTIES
from service import get_balance,get_models

RING=Path('/root/ProLiGen_ring_random_control_20261102')
CONT=Path('/root/ProLiGen_single_property_1000_20261028/run_v2')
def main():
 assert not (ROOT/'inputs/prepared.json').exists()
 RDLogger.DisableLog('rdApp.*')
 for n in ['inputs','source','logs','requests','responses','results','analysis','figures']:(ROOT/n).mkdir(exist_ok=True)
 sources={RING/'inputs/entries500.jsonl',CONT/'inputs/entries1000.jsonl',RING/'source/library.csv',CONT/'inputs/library.csv'}
 oldhash={str(p):sha(p) for p in sources}
 ring=read_jsonl(RING/'inputs/entries500.jsonl');cont=read_jsonl(CONT/'inputs/entries1000.jsonl')
 original=[(RING,e) for e in ring if e['group']=='G']+[(CONT,e) for e in cont if e['group']=='G']
 assert len(original)==375
 targets=[e['chemblid'] for e in ring if e['group']=='G' and e['position']==1];assert len(set(targets))==25
 shutil.copy2(RING/'source/library.csv',ROOT/'source/ring_library.csv');shutil.copy2(CONT/'inputs/library.csv',ROOT/'source/continuous_library.csv')
 for p in list((RING/'source').glob('*.pth'))+list((CONT/'models').glob('*.pth')):
  # Retain only deployed property weights; initializer is not used by generation.
  if 'initializer' in p.name:continue
  target=ROOT/'source'/p.name
  if target.exists():assert sha(target)==sha(p)
  else:shutil.copy2(p,target)
  oldhash[str(p)]=sha(p)
 bases={};prefixes={};pools={};prefix_receipts=[]
 ringlib=pd.read_csv(ROOT/'source/ring_library.csv');conlib=pd.read_csv(ROOT/'source/continuous_library.csv')
 assert len(ringlib)==26108 and len(conlib)==24693
 libs={'ring':ringlib,'logp':conlib,'tpsa':conlib}
 for oldroot,e in original:
  prop=e.get('property','ring');key=f'{prop}_{e["chemblid"]}'
  if e['position']!=1:continue
  pr=properties(e['seed_smiles']);cfg=PROPERTIES[prop];val=pr['RC' if prop=='ring' else prop];center=val+cfg['delta']
  q=dict(e,property=prop,seed_canonical=pr['canonical'],seed_property=val,target_center=center,
    target_lower=center-cfg['tolerance'],target_upper=center+cfg['tolerance'],tolerance=cfg['tolerance'])
  assert len(set(e['exemplars']))==3
  # Frozen common context is taken from one archived G run, identically for both methods and all five new repeats.
  result=read(oldroot/'results'/f'{e["slot_id"]}.json');attempt=result['successful_attempt']
  first=sorted((oldroot/'responses').glob(f'{e["slot_id"]}_a{attempt}_t1_s*.json'))
  first=[p for p in first if not p.stem.endswith('_error') and 'response' in read(p)]
  assert len(first)==1
  req=read(oldroot/'requests'/first[0].name)
  messages=[req['request']['messages'][0]]
  for turn in range(1,4):
   matches=[p for p in (oldroot/'responses').glob(f'{e["slot_id"]}_a{attempt}_t{turn}_s*.json') if not p.stem.endswith('_error') and 'response' in read(p)]
   assert len(matches)==1;p=matches[0];body=read(p)['response'];choice=body['choices'][0]
   assert choice['finish_reason']=='stop' and choice['message']['content'].strip()
   messages.extend([dict(role='user',content=e['prompt'][turn-1]),dict(role='assistant',content=choice['message']['content'])])
   prefix_receipts.append(dict(task=key,turn=turn,path=str(p),sha256=sha(p)))
  # Background must not contain the three property-selected G examples.
  assert all(s not in json.dumps(messages,ensure_ascii=False) for s in e['exemplars'])
  prefix_sha=hashlib.sha256(json.dumps(messages,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  prefixes[key]=messages;bases[key]=dict(q,prefix_key=key,prefix_sha256=prefix_sha)
  lib=libs[prop];column='RingCount' if prop=='ring' else prop
  vals=lib[column].astype(float)
  mask=(vals-center).abs()<=cfg['tolerance']+1e-8
  selected=lib.loc[mask & (lib.canonical!=pr['canonical']),'canonical'].tolist()
  assert len(selected)>=3 and set(e['exemplars']).issubset(set(selected))
  assert len(selected)==len(set(selected))
  pools[key]=selected
 assert len(bases)==len(prefixes)==len(pools)==75
 entries=[]
 for prop in ['ring','logp','tpsa']:
  for target in targets:
   base=bases[f'{prop}_{target}']
   for repeat in range(1,6):
    for method in ['G','CD']:
     entries.append(dict(base,group=method,method='Pro-LiGen G' if method=='G' else 'ChatDrug adapted',position=repeat,
       slot_id=f'{prop}_{target}_{repeat}_{method}',exemplars=list(base['exemplars']) if method=='G' else []))
 rng=random.Random(42);blocks=[(p,t,r) for p in ['ring','logp','tpsa'] for t in targets for r in range(1,6)];rng.shuffle(blocks)
 schedule=[]
 for prop,target,repeat in blocks:
  methods=['G','CD'];rng.shuffle(methods)
  for method in methods:schedule.append(dict(slot_id=f'{prop}_{target}_{repeat}_{method}',property=prop,chemblid=target,position=repeat,
   group=method,order=len(schedule)+1,pilot=(target in targets[:2] and repeat==1)))
 assert len(schedule)==750 and sum(s['pilot'] for s in schedule)==12
 with (ROOT/'inputs/entries750.jsonl').open('w',encoding='utf-8') as f:
  for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
 dump(ROOT/'inputs/contexts75.json',prefixes);dump(ROOT/'inputs/pools75.json',pools)
 dump(ROOT/'inputs/schedule750.json',schedule);dump(ROOT/'inputs/context_provenance.json',prefix_receipts)
 dump(ROOT/'inputs/protected_sources.json',oldhash)
 protocol=dict(created_utc=now(),planned_final_positions=750,conditions=PROPERTIES,targets=targets,repeats=5,methods=['G','CD'],
  input_mode='Same frozen three-exchange target/literature/seed context for both methods; all new proposal calls contemporaneous; no old generated candidates reused.',
  context_repeats='One archived first-repeat G background per task-condition, fixed across five fresh repeats and both methods.',
  G='Original G proposal prompts 4/5, two x three candidates; computed feedback; eligible union best QED with SA/canonical ties.',
  CD='Original first-parsed-candidate rule; initial request five SMILES; C=2 means initial plus at most two revisions; first eligible stops; invalid/unparsed stops and counts as failure.',
  CD_retrieval='Same exact-condition pool; sparse count Morgan radius2 Tanimoto nearest to failed candidate; tie uses original library row order; one reference per revision.',
  CD_adaptations=['Common frozen task context/model service/library/condition evaluator','Ring task and bounded continuous-property targets','Common nonempty single-component carbon/noncopy checks','Durable logging and bounded technical retries'],
  output_policy='Native outputs primary; common candidate replay secondary; no QED selector inserted into ChatDrug.',
  candidate_budget='G requests six raw candidates; CD original initial five plus up to two one-candidate revisions, at most seven requested; actual extracted and evaluated counts retained.',
  primary=dict(endpoints=['S','U'],contrast='G-CD',unit='25 paired task means',family='Holm across 3 conditions x2 endpoints =6 tests',
    sign_flip_draws=100000,bootstrap_draws=10000,random_seed=42),
  secondary=['conditional QED/SA','seed Morgan Tanimoto','candidate replay','actual calls and raw candidate count','validity and failure reasons'],
  pilot_positions=12,pilot_excluded_from_formal=True,pilot_purpose='Technical and input/parse/feedback/selector checks; no requirement that G outperform CD.',
  api=dict(model='deepseek-flash',documented_version='DeepSeek-V4.1-Flash-1017',max_tokens=16384,reasoning_effort='low',thinking='enabled',temperature='omitted',top_p='omitted'),
  no_new_training=True,no_docking=True,no_manuscript_edits=True,no_publication=True,
  authorization='User selected 750 positions and requested execution on 2026-11-08; same previously authorized official DeepSeek service and scientific input types.',
  pause_policy='Atomic request/response/slot records, SQLite WAL; live balance reserve2CNY; HTTP402 pauses; never repeat completed paid calls; unknown in-flight calls require reconciliation.')
 dump(ROOT/'protocol.json',protocol)
 probe=dict(balance=get_balance(),models=get_models(),checked_utc=now(),paid_generation_calls=0,credentials_recorded=False)
 dump(ROOT/'inputs/api_probe.json',probe)
 assert all(sha(p)==v for p,v in oldhash.items())
 dump(ROOT/'inputs/prepared.json',dict(status='passed',entries=750,paired_task_conditions=75,frozen_contexts=75,pilot_positions=12,
  source_libraries={'ring':26108,'continuous':24693},rdkit=rdBase.rdkitVersion,original_sources_unchanged=True))
 print(json.dumps(dict(status='prepared',formal=750,pilot=12,pools=len(pools),balance_cny=probe['balance']['balance_cny'],new_paid_calls=0)))
if __name__=='__main__':main()
