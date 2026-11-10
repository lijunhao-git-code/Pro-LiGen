"""Independent RDKit recomputation and raw-reply audit; never uses API calls."""
from pathlib import Path
import argparse,hashlib,json,re,sqlite3,math
from rdkit import Chem,RDLogger
from rdkit.Chem import Descriptors,QED
from common import ROOT,read,read_jsonl,dump,sha,sa_scorer,now,choose
from comparison import native_first,occurrences,Retrieval

def recompute(smiles,entry,refs):
 mol=Chem.MolFromSmiles(smiles or '')
 if mol is None or not mol.GetNumHeavyAtoms():return dict(valid=False)
 canon=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
 prop={'ring':Descriptors.RingCount,'logp':Descriptors.MolLogP,'tpsa':Descriptors.TPSA}[entry['property']](mol)
 hit=abs(prop-entry['target_center'])<=entry['tolerance']+1e-8
 carbon=any(a.GetAtomicNum()==6 for a in mol.GetAtoms())
 valid_output=(len(Chem.GetMolFrags(mol))==1 and (entry['property']=='ring' or carbon) and canon!=entry['seed_canonical'] and canon not in refs and hit)
 return dict(valid=True,canonical=canon,property_value=prop,property_hit=hit,eligible=valid_output,
  QED=float(QED.qed(mol)),SA_raw=float(sa_scorer().calculateScore(mol)))
def run(pilot=False,archive_mode=False):
 RDLogger.DisableLog('rdApp.*');base=ROOT/'pilot' if pilot else ROOT
 frozen=read(ROOT/'inputs/generation_locked.json');assert all(sha(ROOT/n)==value for n,value in frozen['files'].items())
 entries={e['slot_id']:e for e in read_jsonl(ROOT/'inputs/entries750.jsonl')};contexts=read(ROOT/'inputs/contexts75.json')
 schedule=read(ROOT/'inputs/schedule750.json');expected={s['slot_id'] for s in schedule if s['pilot'] or not pilot}
 results=[read(p) for p in sorted((base/'results').glob('*.json'))];assert {r['slot_id'] for r in results}==expected,'Incomplete results'
 retrieval=Retrieval();errors=[];rows_checked=0;selected_checked=0;retrievals=0;records=[];discarded_technical_responses=[]
 result_map={r['slot_id']:r for r in results}
 db=sqlite3.connect('file:'+str(base/'logs/usage.sqlite')+'?mode=ro',uri=True)
 api_states=dict(db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state').fetchall())
 assert not any(k in api_states for k in ['started','fatal_error','balance_rejected'])
 for r in results:
  entry=entries[r['slot_id']]
  assert r.get('common_context_sha256',entry['prefix_sha256'])==entry['prefix_sha256']
  for c in r['candidates']:
   refs=entry['exemplars'] if r['group']=='G' else next(t['provided_references'] for t in r['trace'] if t['turn']==c['turn'])
   fresh=recompute(c.get('smiles'),entry,refs)
   assert fresh['valid']==c['valid'],(r['slot_id'],'valid')
   if c['valid']:
    for k in ['canonical','property_hit','eligible']:assert fresh[k]==c[k],(r['slot_id'],k)
    for k in ['property_value','QED','SA_raw']:assert abs(fresh[k]-c[k])<1e-8,(r['slot_id'],k)
   rows_checked+=1
  if not r['technical_complete']:continue
  if r['group']=='G':
   selected=choose(r['candidates']);assert len(r['candidates'])==6 and len(r['trace'])==2
   for t in r['trace']:
    record=db.execute("SELECT response_path FROM calls WHERE id=? AND state='ok'",(f'{r["slot_id"]}_a{r["successful_attempt"]}_t{t["turn"]}',)).fetchone();assert record
    text=read(base/record[0])['response']['choices'][0]['message']['content']
    tags=re.findall(r'<smiles>\s*(.*?)\s*</smiles>',text,flags=re.I|re.S)
    expected_smiles=[tags[i].strip() if i<len(tags) else None for i in range(3)]
    assert [c['smiles'] for c in r['candidates'] if c['turn']==t['turn']]==expected_smiles
  else:
   native=[c for c in r['candidates'] if c.get('native_evaluated')]
   selected=next((c for c in native if c['eligible']),None)
   assert len(r['trace'])<=3
   if selected:assert selected['turn']==r['trace'][-1]['turn'],'Native workflow must stop on first success'
   for t in r['trace']:
    record=db.execute("SELECT response_path FROM calls WHERE id=? AND state='ok'",(f'{r["slot_id"]}_a{r["successful_attempt"]}_t{t["turn"]}',)).fetchone();assert record
    response=base/record[0]
    text=read(response)['response']['choices'][0]['message']['content']
    last=t['provided_references'][-1] if t['provided_references'] else None
    first=native_first(text,entry['seed_smiles'],last)
    observed=next((c for c in r['candidates'] if c['turn']==t['turn'] and c.get('native_evaluated')),None)
    assert [c['smiles'] for c in r['candidates'] if c['turn']==t['turn']]==occurrences(text)
    assert (observed['smiles'] if observed else None)==first
    if 'retrieved_reference' in t:
     ref,score=retrieval.closest(entry['prefix_key'],observed['canonical'])
     assert ref==t['retrieved_reference'] and abs(score-t['retrieval_tanimoto'])<1e-12;retrievals+=1
  assert bool(selected)==r['ok'] and (selected['canonical'] if selected else None)==r['canonical']
  if selected:
   for k in ['canonical','property_value','QED','SA_raw']:assert selected[k]==r['selected_candidate'][k]
  if selected:assert selected['eligible'];selected_checked+=1
  records.append(dict(slot_id=r['slot_id'],property=r['property'],group=r['group'],ok=r['ok'],status=r['status'],
   candidate_count=len(r['candidates']),native_turns=len(r['trace']),selected_property=selected['property_value'] if selected else None,
   target_lower=entry['target_lower'],target_upper=entry['target_upper']))
 for call,request_sha,response_path in db.execute("SELECT id,request_sha,response_path FROM calls WHERE state='ok'"):
  response=base/response_path;body=read(response)
  req=read(base/'requests'/response.name)
  assert req['request_sha256']==request_sha==hashlib.sha256(json.dumps(req['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
  entry=entries[req['slot_id']];prefix=contexts[entry['prefix_key']]
  assert req['request']['messages'][:len(prefix)]==prefix
  assert body['response']['model']=='deepseek-flash'
  choice=body['response']['choices'][0]
  if choice.get('finish_reason')!='stop' or not (choice['message'].get('content') or '').strip():
   result=result_map[req['slot_id']]
   assert req['attempt']<result['successful_attempt']
   assert any(a['attempt']==req['attempt'] and a['reason']=='empty_or_truncated_response' for a in result['earlier_attempts'])
   discarded_technical_responses.append(dict(call_id=call,finish_reason=choice.get('finish_reason'),not_used_for_final_output=True))
 db.close()
 if archive_mode:
  manifest=read(ROOT/'file_manifest.json')['files']
  protected=[n for n in manifest if n.startswith(('inputs/','source/','requests/','responses/','results/','code/')) or n=='protocol.json']
  assert all(sha(ROOT/n)==manifest[n]['sha256'] for n in protected)
  source_integrity=None
 else:
  source_integrity=all(sha(p)==v for p,v in read(ROOT/'inputs/protected_sources.json').items());assert source_integrity
 report=dict(status='passed',checked_utc=now(),mode='pilot' if pilot else 'full',results=len(results),candidate_records_recomputed=rows_checked,
  delivered_selections_verified=selected_checked,retrieval_events_verified=retrievals,api_states=api_states,
  discarded_technical_responses=discarded_technical_responses,retried_positions=sum(bool(r.get('earlier_attempts')) for r in results),
  original_sources_unchanged=source_integrity,frozen_generation_code_and_inputs_unchanged=True,archive_mode=archive_mode,all_outputs_match_raw_replies=True,all_properties_independently_recomputed=True,
  allow_full_run=bool(pilot and len(results)==12 and all(r['technical_complete'] for r in results)),
  pilot_outcomes_not_used_to_tune_methods=True,record_summary=records)
 output=ROOT/'analysis'/('pilot_review.json' if pilot else 'full_audit.json');dump(output,report)
 print(json.dumps({k:v for k,v in report.items() if k!='record_summary'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--archive-mode',action='store_true');args=ap.parse_args();run(args.pilot,args.archive_mode)
