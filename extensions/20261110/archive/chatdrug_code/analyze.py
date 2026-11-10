"""Native outputs, candidate replay and paired task inference from actual saved responses."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import argparse,json,sqlite3,hashlib
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from common import ROOT,read,read_jsonl,dump,now,sha,choose

def compare(x,seed=42):
 x=np.asarray(x,float);rng=np.random.default_rng(seed);b=100000
 observed=float(x.mean());permuted=[]
 for _ in range(10):permuted.extend((rng.choice([-1,1],size=(b//10,len(x)))*x).mean(1))
 p=(1+np.count_nonzero(np.abs(permuted)>=abs(observed)-1e-12))/(b+1)
 boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
 return dict(difference=observed,ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),p_raw=float(p),tasks=len(x))
def holm(values):
 values=np.asarray(values);order=np.argsort(values);ans=np.empty(len(values));previous=0.
 for i,j in enumerate(order):previous=max(previous,min(1.,(len(values)-i)*values[j]));ans[j]=previous
 return ans

def main():
 RDLogger.DisableLog('rdApp.*');(ROOT/'analysis').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
 assert read(ROOT/'analysis/full_audit.json')['status']=='passed'
 entries={e['slot_id']:e for e in read_jsonl(ROOT/'inputs/entries750.jsonl')};results=[read(p) for p in sorted((ROOT/'results').glob('*.json'))]
 assert len(results)==750 and {r['slot_id'] for r in results}==set(entries)
 fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048);fps={}
 def fp(s):
  if s not in fps:fps[s]=fpgen.GetFingerprint(Chem.MolFromSmiles(s))
  return fps[s]
 rows=[];candidates=[];replays=[];frontiers=[]
 for r in results:
  e=entries[r['slot_id']];c=r['selected_candidate'];base={k:r[k] for k in ['slot_id','property','chemblid','position','group']}
  row=dict(base,ok=int(r['ok']),status=r['status'],technical_complete=r['technical_complete'],canonical=r['canonical'],
   QED=c['QED'] if c else None,U=c['QED'] if c else 0.,SA_raw=c['SA_raw'] if c else None,
   tanimoto=float(DataStructs.TanimotoSimilarity(fp(e['seed_smiles']),fp(c['canonical']))) if c else None,
   property_value=c['property_value'] if c else None,seed_property=e['seed_property'],target_lower=e['target_lower'],target_upper=e['target_upper'],
   delta_property=c['delta_property'] if c else None,MW=c['MW'] if c else None,RC=c['RC'] if c else None,
   recorded_candidates=len(r['candidates']),valid_candidates=sum(x['valid'] for x in r['candidates']),eligible_candidates=sum(x['eligible'] for x in r['candidates']),
   proposal_replies=len(r['trace']),response_source=f'results/{r["slot_id"]}.json',prefix_sha256=e['prefix_sha256'])
  rows.append(row)
  for x in r['candidates']:candidates.append(dict(base,**x,response_source=f'results/{r["slot_id"]}.json'))
  ordered=sorted(r['candidates'],key=lambda x:(x['turn'],x['candidate_position']))
  eligible=[x for x in ordered if x['eligible']];unique={x['canonical']:x for x in eligible}
  for policy in ['native','first_eligible','best_QED','expected_uniform_unique_eligible']:
   if policy=='native':s=c;q=s['QED'] if s else 0.;ok=bool(s)
   elif policy=='expected_uniform_unique_eligible':ok=bool(unique);q=float(np.mean([x['QED'] for x in unique.values()])) if unique else 0.
   else:s=choose(ordered,policy);ok=bool(s);q=s['QED'] if s else 0.
   replays.append(dict(base,policy=policy,S=float(ok),U=q))
  for k in range(1,7):
   first=ordered[:k];best=choose(first)
   frontiers.append(dict(base,k=k,S=float(best is not None),U=best['QED'] if best else 0.))
 frame=pd.DataFrame(rows).sort_values(['property','chemblid','position','group']);frame.to_csv(ROOT/'analysis/final_outputs750.csv',index=False)
 pools=read(ROOT/'inputs/pools75.json')
 pd.DataFrame([dict(task_condition=k,pool_size=len(v),canonical_sequence_sha256=hashlib.sha256('\n'.join(v).encode()).hexdigest()) for k,v in pools.items()]).to_csv(ROOT/'analysis/reference_pools75.csv',index=False)
 reference_events=[]
 for r in results:
  e=entries[r['slot_id']]
  if r['group']=='G':
   for rank,s in enumerate(e['exemplars'],1):reference_events.append(dict(slot_id=r['slot_id'],property=r['property'],group='G',rank=rank,turn=4,smiles=s,in_common_pool=s in pools[e['prefix_key']]))
  else:
   for t in r['trace']:
    if 'retrieved_reference' in t:
     s=t['retrieved_reference'];reference_events.append(dict(slot_id=r['slot_id'],property=r['property'],group='CD',turn=t['turn']+1,smiles=s,in_common_pool=s in pools[e['prefix_key']],retrieval_tanimoto=t['retrieval_tanimoto']))
 assert all(x['in_common_pool'] for x in reference_events)
 pd.DataFrame(reference_events).to_csv(ROOT/'analysis/provided_reference_events.csv',index=False)
 pd.DataFrame(candidates).to_csv(ROOT/'analysis/all_candidate_records.csv',index=False)
 tasks=frame.groupby(['property','chemblid','group'],as_index=False).agg(S=('ok','mean'),U=('U','mean'),
  QED=('QED','mean'),SA_raw=('SA_raw','mean'),tanimoto=('tanimoto','mean'),delivered=('ok','sum'),n=('slot_id','size'))
 assert len(tasks)==150 and tasks.n.eq(5).all();tasks.to_csv(ROOT/'analysis/task_means150.csv',index=False)
 summary=[];tests=[];difference_rows=[]
 def nullable(value):return None if pd.isna(value) else float(value)
 for prop in ['ring','logp','tpsa']:
  for group in ['G','CD']:
   f=frame[(frame.property==prop)&(frame.group==group)];t=tasks[(tasks.property==prop)&(tasks.group==group)]
   rng=np.random.default_rng(42);idx=rng.integers(0,25,(10000,25))
   item=dict(property=prop,group=group,planned=len(f),delivered=int(f.ok.sum()),S=float(t.S.mean()),U=float(t.U.mean()),
    conditional_QED=nullable(t.QED.mean()),conditional_SA_raw=nullable(t.SA_raw.mean()),conditional_tanimoto=nullable(t.tanimoto.mean()),
    tasks_with_delivery=int(t.delivered.gt(0).sum()),raw_candidates=int(f.recorded_candidates.sum()),valid_candidates=int(f.valid_candidates.sum()),
    eligible_candidates=int(f.eligible_candidates.sum()),mean_proposal_replies=float(f.proposal_replies.mean()))
   for metric in ['S','U']:
    boot=t[metric].to_numpy()[idx].mean(1);item[metric+'_ci_low']=float(np.quantile(boot,.025));item[metric+'_ci_high']=float(np.quantile(boot,.975))
   summary.append(item)
  g=tasks[(tasks.property==prop)&(tasks.group=='G')].set_index('chemblid');cd=tasks[(tasks.property==prop)&(tasks.group=='CD')].set_index('chemblid')
  for target in g.index:difference_rows.append(dict(property=prop,chemblid=target,S_difference=float(g.loc[target,'S']-cd.loc[target,'S']),U_difference=float(g.loc[target,'U']-cd.loc[target,'U'])))
  for metric in ['S','U']:tests.append(dict(property=prop,metric=metric,contrast='G-CD',**compare(g[metric].sort_index()-cd[metric].sort_index())))
 adjusted=holm([x['p_raw'] for x in tests])
 for x,p in zip(tests,adjusted):x['p_Holm']=float(p)
 sm=pd.DataFrame(summary);st=pd.DataFrame(tests);sm.to_csv(ROOT/'analysis/native_summary6.csv',index=False);st.to_csv(ROOT/'analysis/paired_tests6.csv',index=False)
 pd.DataFrame(difference_rows).to_csv(ROOT/'analysis/task_differences75.csv',index=False)
 pd.DataFrame(replays).to_csv(ROOT/'analysis/candidate_replay_slots.csv',index=False)
 rep=pd.DataFrame(replays).groupby(['property','group','policy'],as_index=False)[['S','U']].mean();rep.to_csv(ROOT/'analysis/candidate_replay_summary.csv',index=False)
 pd.DataFrame(frontiers).to_csv(ROOT/'analysis/candidate_frontier_slots.csv',index=False)
 frontier=pd.DataFrame(frontiers).groupby(['property','group','k'],as_index=False)[['S','U']].mean();frontier.to_csv(ROOT/'analysis/candidate_frontier_summary.csv',index=False)
 frame.groupby(['property','group','status']).size().reset_index(name='positions').to_csv(ROOT/'analysis/outcome_reasons.csv',index=False)
 usage=[]
 for cohort,base in [('formal',ROOT),('pilot',ROOT/'pilot')]:
  db=sqlite3.connect('file:'+str(base/'logs/usage.sqlite')+'?mode=ro',uri=True)
  db.row_factory=sqlite3.Row
  for d in db.execute('SELECT * FROM calls'):
   d=dict(d);u=json.loads(d.pop('usage'));rr=read(base/d['response_path']) if d['state']=='ok' else {}
   usage.append(dict(cohort=cohort,**d,prompt_tokens=u.get('prompt_tokens',0),completion_tokens=u.get('completion_tokens',0),
    elapsed_seconds=rr.get('elapsed_seconds'),started_utc=rr.get('started_utc'),finished_utc=rr.get('finished_utc'),
    model=rr.get('response',{}).get('model'),system_fingerprint=rr.get('response',{}).get('system_fingerprint')))
  db.close()
 us=pd.DataFrame(usage);us.to_csv(ROOT/'analysis/API_usage_by_call.csv',index=False)
 api=[]
 for cohort,f in us.groupby('cohort'):
  success=f[f.state=='ok'];starts=[datetime.fromisoformat(s) for s in success.started_utc];ends=[datetime.fromisoformat(s) for s in success.finished_utc]
  api.append(dict(cohort=cohort,call_identifiers=len(f),submissions=int(f.submission.sum()),successful_calls=len(success),prompt_tokens=int(f.prompt_tokens.sum()),completion_tokens=int(f.completion_tokens.sum()),
   archived_ratecard_estimate_cny=float(f.charge_estimate.sum()),archived_peak_rate_estimate_cny=float(f.charge_peak.sum()),
   median_call_seconds=float(success.elapsed_seconds.median()),wall_clock_seconds=(max(ends)-min(starts)).total_seconds(),
   response_models=sorted(success.model.dropna().unique()),system_fingerprints=sorted(success.system_fingerprint.dropna().unique())))
 dump(ROOT/'analysis/API_resource_summary.json',api)
 # Standard static scientific plots; plotted source values are saved above.
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':240})
 colors={'G':'#315D87','CD':'#D69045'};names={'ring':'One fewer ring','logp':'Lower LogP','tpsa':'Lower TPSA'}
 fig,axes=plt.subplots(2,3,figsize=(10,5.7))
 for j,prop in enumerate(['ring','logp','tpsa']):
  for i,metric in enumerate(['S','U']):
   ax=axes[i,j];data=sm[sm.property==prop].set_index('group').loc[['G','CD']]
   y=data[metric].to_numpy();err=np.stack([y-data[metric+'_ci_low'],data[metric+'_ci_high']-y])
   ax.bar([0,1],y,color=[colors[x] for x in ['G','CD']],width=.55,yerr=err,capsize=3)
   ax.set_xticks([0,1],['Pro-LiGen G','ChatDrug adapted']);ax.set_ylim(0,1.12 if metric=='S' else 1.)
   ax.set_title(names[prop] if i==0 else '',pad=27);ax.set_ylabel('Qualified delivery' if metric=='S' else 'QED yield per run')
   for k,val in enumerate(y):ax.text(k,float(data[metric+'_ci_high'].iloc[k])+.025,f'{val:.3f}',ha='center',fontsize=9)
   p=st[(st.property==prop)&(st.metric==metric)].p_Holm.iloc[0];ax.text(.5,1.025,f'paired Holm p = {p:.3g}',transform=ax.transAxes,ha='center',fontsize=8)
 fig.tight_layout(h_pad=2.2);fig.savefig(ROOT/'figures/native_comparison.png');fig.savefig(ROOT/'figures/native_comparison.pdf');plt.close(fig)
 fig,axes=plt.subplots(2,3,figsize=(10,5.7))
 for j,prop in enumerate(['ring','logp','tpsa']):
  for i,metric in enumerate(['S','U']):
   ax=axes[i,j]
   for group in ['G','CD']:
    d=frontier[(frontier.property==prop)&(frontier.group==group)];ax.plot(d.k,d[metric],marker='o',color=colors[group],label='Pro-LiGen G' if group=='G' else 'ChatDrug adapted')
   ax.set_xticks(range(1,7));ax.set_ylim(0,1.03);ax.set_title(names[prop] if i==0 else '');ax.set_xlabel('First k recorded candidate positions');ax.set_ylabel('Qualified delivery' if metric=='S' else 'QED yield');
 axes[0,0].legend(frameon=False,fontsize=8);fig.tight_layout();fig.savefig(ROOT/'figures/common_selector_frontier.png');plt.close(fig)
 out=dict(status='complete',created_utc=now(),formal_positions=750,pilot_positions=12,task_units=25,summary=summary,paired_tests=tests,
  resource_use=api,statistics='G-CD; two-sided paired task sign flips; unadjusted task bootstrap95CI; Holm6 primary tests',
  common_context='Archived background fixed identically across methods and repeats; all proposal calls newly generated.',
  cost_scope='Costs are usage estimates with archived CNY ratecard, not account recharge totals; frozen-background acquisition cost excluded.',
  actual_raw_budget='G requests6; ChatDrug may request5+1+1 but native parser evaluates first only; see candidate frontier and replay diagnostics.',
  native_QED_scope='Whole-workflow output quality includes native selector effects; use replay to interpret candidate-generation differences.',
  paper_edited=False,docking_jobs=0,training_jobs=0)
 dump(ROOT/'analysis/final_summary.json',out)
 print(json.dumps(dict(status='complete',formal_positions=750,summary=summary,tests=tests,resource=api)))
if __name__=='__main__':main()
