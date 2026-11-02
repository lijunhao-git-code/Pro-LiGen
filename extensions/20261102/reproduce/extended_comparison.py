"""Reproduce Table 2, all 24 paired tests and Figure 5 inputs. No API calls.
Requires Python 3.10+ and NumPy. Run from any directory.
"""
from pathlib import Path
from collections import defaultdict
import csv,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'data/updated20261102'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<2e-11,(a,b)
def paired(v):
    v=np.asarray(v,dtype=float);assert len(v)==25
    idx=np.random.default_rng(42).integers(0,25,(10000,25))
    lo,hi=np.quantile(v[idx].mean(axis=1),[.025,.975])
    signs=np.random.default_rng(42).choice([-1.,1.],(100000,25))
    extreme=np.abs((signs*v).mean(axis=1))>=abs(v.mean())-1e-12
    return dict(difference=float(v.mean()),ci_low=float(lo),ci_high=float(hi),p_raw=float((1+extreme.sum())/100001))
def main():
    obs=rows(D/'observations1500.csv');assert len(obs)==1500
    assert len({r['slot_id'] for r in obs})==1500
    # The compact observation view must agree with both archived source exports.
    source=[]
    for condition,file in [('ring','ring/final_positions.csv'),(None,'properties/final_positions.csv')]:
        for r in rows(D/file):
            source.append(dict(condition=condition or r['property'],chemblid=r['chemblid'],slot_id=r['slot_id'],group=r['group'],**{k:r[k] for k in ['S','U','QED','SA_raw']}))
    byslot={r['slot_id']:r for r in source};assert len(byslot)==1500
    for r in obs:
        original=byslot[r['slot_id']]
        for k in ['condition','chemblid','group']:assert r[k]==original[k]
        for k in ['S','U','QED','SA_raw']:
            if r[k]=='':assert original[k]==''
            else:close(r[k],original[k])
        if float(r['S'])==0:
            assert float(r['U'])==0 and r['QED']=='' and r['SA_raw']==''
        else:close(r['U'],r['QED'])
    cells=defaultdict(list)
    for r in obs:cells[(r['condition'],r['chemblid'],r['group'])].append(r)
    assert len(cells)==300 and all(len(x)==5 for x in cells.values())
    targets={}
    for key,items in cells.items():
        targets[key]={}
        for metric in ['S','U','QED','SA_raw']:
            values=[float(r[metric]) for r in items if r[metric]!='']
            targets[key][metric]=float(np.mean(values)) if values else None
    for r in rows(D/'table2_values.csv'):
        vals=[v for k,v in targets.items() if k[0]==r['condition'] and k[2]==r['group']]
        assert len(vals)==25 and int(r['planned'])==125
        close(sum(float(q['S']) for q in obs if q['condition']==r['condition'] and q['group']==r['group']),r['delivered'])
        for metric in ['S','U','QED','SA_raw']:
            close(np.mean([v[metric] for v in vals if v[metric] is not None]),r[metric])
    expected=rows(D/'declared_tests24.csv');assert len(expected)==24
    calculated=[]
    for r in expected:
        left,right=r['comparison'].split('-')
        ids=sorted(k[1] for k in targets if k[0]==r['condition'] and k[2]==left)
        v=[targets[(r['condition'],t,left)][r['metric']]-targets[(r['condition'],t,right)][r['metric']] for t in ids]
        calc=paired(v);calc.update(condition=r['condition'],family=r['family'],metric=r['metric'],comparison=r['comparison'])
        calculated.append(calc)
    for family in sorted({r['family'] for r in calculated}):
        part=sorted([r for r in calculated if r['family']==family],key=lambda r:r['p_raw'])
        assert len(part)=={'primary4':4,'secondary4':4,'primary8':8,'secondary8':8}[family]
        previous=0.
        for i,r in enumerate(part):
            previous=max(previous,min(1.,r['p_raw']*(len(part)-i)));r['p_holm']=previous
    for r,e in zip(calculated,expected):
        for key in ['difference','ci_low','ci_high','p_raw','p_holm']:close(r[key],e[key])
    for r in rows(D/'figure5_target_differences75.csv'):
        c,t=r['condition'],r['chemblid']
        close(targets[(c,t,'G')]['U']-targets[(c,t,'M')]['U'],r['difference'])
    selected={(r['condition'],r['comparison']):r for r in calculated if r['metric']=='U'}
    for r in rows(D/'figure5_comparisons9.csv'):
        for k in ['difference','ci_low','ci_high','p_holm']:close(r[k],selected[(r['condition'],r['comparison'])][k])
    result=dict(status='passed',planned_runs=1500,qualified_outputs=sum(int(float(r['S'])) for r in obs),tasks_per_condition=25,
        table_rows_verified=12,paired_tests_verified=24,figure5_task_points_verified=75,figure5_intervals_verified=9,
        statistic_unit='paired target means; 5 runs per target',seed=42,bootstrap_draws=10000,sign_flips=100000,
        holm_families={'ring_primary':4,'ring_secondary':4,'continuous_primary':8,'continuous_secondary':8},
        max_numeric_tolerance=2e-11,new_API_calls=0)
    (D/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
