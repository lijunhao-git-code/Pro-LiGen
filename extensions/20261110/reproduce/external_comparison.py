"""Verify Study D from frozen scientific records, without API calls.
Python 3.10+ and NumPy. Run from any directory. Does not import generation code.
"""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/external20261108'
def rows(name):
 with (D/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<2e-11,(a,b)
def paired(x):
 x=np.asarray(x,float);rng=np.random.default_rng(42);permuted=[]
 for _ in range(10):permuted.extend((rng.choice([-1,1],size=(10000,len(x)))*x).mean(1))
 p=(1+np.count_nonzero(np.abs(permuted)>=abs(x.mean())-1e-12))/100001
 boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
 lo,hi=np.quantile(boot,[.025,.975])
 return dict(difference=float(x.mean()),ci_low=float(lo),ci_high=float(hi),p_raw=float(p))
def main():
 manifest=json.loads((D/'source_manifest.json').read_text(encoding='utf-8'))
 for name,h in manifest['files'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h,name
 obs=rows('final_outputs750.csv');assert len(obs)==750
 assert len({r['slot_id'] for r in obs})==750
 cells=defaultdict(list)
 selected={}
 for r in obs:
  record=json.loads((D/'results'/f"{r['slot_id']}.json").read_text(encoding='utf-8'))
  assert int(record['ok'])==int(r['ok']) and record['canonical']==(r['canonical'] or None)
  c=record['selected_candidate'];selected[r['slot_id']]=c
  if c:
   for k in ['QED','SA_raw','property_value']:close(c[k],r[k])
   close(r['U'],r['QED'])
  else:assert float(r['U'])==0 and r['QED']==''
  cells[(r['property'],r['chemblid'],r['group'])].append(r)
 assert len(cells)==150 and all(len(v)==5 for v in cells.values())
 tasks={}
 for key,items in cells.items():
  tasks[key]={}
  for metric,field in [('S','ok'),('U','U'),('QED','QED'),('SA_raw','SA_raw'),('tanimoto','tanimoto')]:
   values=[float(r[field]) for r in items if r[field]!='']
   tasks[key][metric]=float(np.mean(values)) if values else None
 for t in rows('task_means150.csv'):
  for m in ['S','U','QED','SA_raw','tanimoto']:
   val=tasks[(t['property'],t['chemblid'],t['group'])][m]
   if val is None:assert t[m]==''
   else:close(val,t[m])
 summary=rows('native_summary6.csv');assert len(summary)==6
 for r in summary:
  part=[o for o in obs if (o['property'],o['group'])==(r['property'],r['group'])]
  t=[v for k,v in tasks.items() if k[0]==r['property'] and k[2]==r['group']]
  assert len(part)==125 and len(t)==25 and sum(int(o['ok']) for o in part)==int(r['delivered'])
  for m,dest in [('S','S'),('U','U'),('QED','conditional_QED'),('SA_raw','conditional_SA_raw'),('tanimoto','conditional_tanimoto')]:
   close(np.mean([v[m] for v in t if v[m] is not None]),r[dest])
  assert sum(v['QED'] is not None for v in t)==int(r['tasks_with_delivery'])
 tests=rows('paired_tests6.csv');calculated=[]
 for r in tests:
  ids=sorted(k[1] for k in tasks if k[0]==r['property'] and k[2]=='G')
  x=[tasks[(r['property'],t,'G')][r['metric']]-tasks[(r['property'],t,'CD')][r['metric']] for t in ids]
  assert len(x)==25;calculated.append(paired(x))
 order=np.argsort([r['p_raw'] for r in calculated]);previous=0.
 for i,j in enumerate(order):
  previous=max(previous,min(1.,(6-i)*calculated[j]['p_raw']));calculated[j]['p_Holm']=previous
 for expected,calc in zip(tests,calculated):
  for k,v in calc.items():close(v,expected[k])
 # Rebuild saved-candidate selection summaries, retaining the full original precision.
 candidates=rows('all_candidate_records.csv');assert len(candidates)==4390
 byslot=defaultdict(list)
 for c in candidates:byslot[c['slot_id']].append(c)
 replay=defaultdict(list);frontier=defaultdict(list)
 def best(items):
  valid=[x for x in items if x['eligible']=='True']
  # Same ordering as frozen common.choose: highest QED, lower SA, canonical tie.
  return sorted(valid,key=lambda c:(-float(c['QED']),float(c['SA_raw']),c['canonical']))[0] if valid else None
 def value(c):return (float(c is not None),float(c['QED']) if c else 0.)
 for r in obs:
  cs=sorted(byslot[r['slot_id']],key=lambda c:(int(c['turn']),int(c['candidate_position'])))
  eligible=[c for c in cs if c['eligible']=='True'];unique={c['canonical']:c for c in eligible}
  pair=(r['property'],r['group'])
  replay[(*pair,'native')].append((float(r['ok']),float(r['U'])))
  replay[(*pair,'first_eligible')].append(value(eligible[0] if eligible else None))
  replay[(*pair,'best_QED')].append(value(best(cs)))
  replay[(*pair,'expected_uniform_unique_eligible')].append((float(bool(unique)),float(np.mean([float(c['QED']) for c in unique.values()])) if unique else 0.))
  for k in range(1,7):frontier[(*pair,k)].append(value(best(cs[:k])))
 for r in rows('candidate_replay_summary.csv'):
  values=np.mean(replay[(r['property'],r['group'],r['policy'])],axis=0)
  close(values[0],r['S']);close(values[1],r['U'])
 for r in rows('candidate_frontier_summary.csv'):
  values=np.mean(frontier[(r['property'],r['group'],int(r['k']))],axis=0)
  close(values[0],r['S']);close(values[1],r['U'])
 # Check every displayed main-table number against the regenerated aggregates.
 tex=(ROOT/'tables/table_external_comparison.tex').read_text(encoding='utf-8')
 for r in summary:
  label='Pro-LiGen' if r['group']=='G' else 'ChatDrug (adapted)'
  line=f"{label} & {r['delivered']}/125 & {100*float(r['S']):.1f} & {float(r['U']):.4f} & {float(r['conditional_QED']):.4f} & {float(r['conditional_SA_raw']):.4f} & {float(r['conditional_tanimoto']):.4f}"
  assert line in tex,line
 result=dict(status='passed',formal_positions=750,qualified_outputs=sum(int(r['ok']) for r in obs),task_means=150,main_table_rows=6,paired_tests=6,candidate_records=4390,replay_cells=24,frontier_cells=36,API_calls=0,training_jobs=0,docking_jobs=0)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
