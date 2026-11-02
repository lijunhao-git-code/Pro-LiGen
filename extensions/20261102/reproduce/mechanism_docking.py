"""Verify batch-2 docking and selection results from frozen observation/proposal records.
Python >=3.10 and NumPy. No API, training or docking execution.
"""
from pathlib import Path
from collections import defaultdict
import csv,hashlib,json
import numpy as np
P=Path(__file__).resolve().parents[1];D=P/'data/mechanism_docking20261102'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def equal(a,b):assert abs(float(a)-float(b))<2e-11,(a,b)
def boolean(x):assert str(x).lower() in ['true','false','1','0'];return str(x).lower() in ['true','1']
def mean(xs):return float(np.mean([float(x) for x in xs]))
def interval(xs):
    v=np.array(xs,dtype=float);i=np.random.default_rng(42).integers(0,len(v),(10000,len(v)))
    return np.quantile(v[i].mean(axis=1),[.025,.975])
def main():
    manifest=json.loads((D/'source_manifest.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((D/n).read_bytes()).hexdigest()==r['sha256'] for n,r in manifest.items())
    obs=rows(D/'docking_observations759.csv');assert len(obs)==759
    assert len({r['observation_id'] for r in obs})==759
    originals={}
    for c,file in [('ring','ring/docking_observations.csv'),(None,'properties/docking_observations.csv')]:
        for r in rows(D/file):originals[r['observation_id']]=dict(r,condition=c or r['property'])
    assert len(originals)==759
    audit={r['job_id']:r for r in rows(D/'ring/job_audit.csv')}
    seeds={r['chemblid']:r for r in rows(D/'properties/seed_scores.csv')}
    assert len(seeds)==11 and len(audit)==198
    final={r['slot_id']:r for r in rows(P/'data/updated20261102/ring/final_positions.csv')+rows(P/'data/updated20261102/properties/final_positions.csv')}
    cells=defaultdict(list)
    for r in obs:
        raw=originals[r['observation_id']];ok=int(r['docking_ok'])
        assert r['condition']==raw['condition'] and r['kind']==raw['kind'] and r['chemblid']==raw['chemblid']
        assert r['group']==(raw['display_group'] if r['condition']=='ring' else raw['group'])
        assert ok==int(float(raw['docking_ok']))
        equal(r['seed_vina'],seeds[r['chemblid']]['vina'])
        if r['kind']=='generated':
            f=final[r['slot_id']];assert ok==int(float(f['S']))
            assert r['canonical']==f['canonical']
        if not ok:
            assert all(r[k]=='' for k in ['vina','QED','delta_vina','delta_QED','Tanimoto']);continue
        for k in ['vina','QED','seed_vina','seed_QED','delta_vina','delta_QED']:equal(r[k],raw[k])
        equal(float(r['vina'])-float(r['seed_vina']),r['delta_vina'])
        equal(float(r['QED'])-float(r['seed_QED']),r['delta_QED'])
        if r['condition']=='ring':
            a=audit[r['job_id']];assert a['canonical']==r['canonical']
            equal(r['vina'],a['vina']);equal(r['vina'],a['first_pose_score'])
            for k in ['prepared_identity_and_input_stereo_match','pose_reconstructed_identity_and_input_stereo_match','pose_3d_input_stereo_match']:assert boolean(a[k])
        cells[(r['condition'],r['group'],r['chemblid'])].append(r)
    target={k:{m:mean(r[m] for r in v) for m in ['vina','seed_vina','QED','seed_QED','delta_vina','delta_QED','Tanimoto']} for k,v in cells.items()}
    assert len(target)==231
    for c in ['ring','logp','tpsa']:
        for g in ['T','M','G','R','LIB_NN1','LIB_QED3','LIB_QED6']:
            assert {k[2] for k in target if k[:2]==(c,g)}==set(seeds)
    for r in rows(D/'docking_target_means231.csv'):
        for m in ['vina','seed_vina','QED','seed_QED','delta_vina','delta_QED','Tanimoto']:equal(r[m],target[(r['condition'],r['group'],r['chemblid'])][m])
    summary={}
    for r in rows(D/'docking_group_summary21.csv'):
        c,g=r['condition'],r['group'];ids=sorted(k[2] for k in target if k[:2]==(c,g));a={m:mean(target[(c,g,t)][m] for t in ids) for m in ['vina','delta_vina','QED','seed_QED','delta_QED','Tanimoto']}
        for m in ['delta_vina','delta_QED','Tanimoto']:
            lo,hi=interval([target[(c,g,t)][m] for t in ids]);a[m+'_ci_low']=lo;a[m+'_ci_high']=hi
        for m in a:equal(r[m],a[m])
        subset=[o for o in obs if o['condition']==c and o['group']==g]
        assert len(subset)==int(r['planned']) and sum(int(o['docking_ok']) for o in subset)==int(r['docked'])
        summary[(c,g)]=r
    for c,file in [('ring','ring/docking_group_summary.csv'),(None,'properties/docking_group_summary.csv')]:
        for r in rows(D/file):
            a=summary[(c or r['property'],r['group'])]
            for m in ['delta_vina','delta_QED']:equal(a[m],r[m])
            assert int(a['docked'])==int(r['docked'])
    for r in rows(D/'table3_values12.csv')+rows(D/'figure6_means21.csv'):
        for m in ['delta_vina','delta_QED','Tanimoto','delta_vina_ci_low','delta_vina_ci_high']:equal(r[m],summary[(r['condition'],r['group'])][m])
    for r in rows(D/'figure6_target_points132.csv'):
        for m in ['delta_vina','delta_QED']:equal(r[m],target[(r['condition'],r['group'],r['chemblid'])][m])
    # Independently reconstruct the five final-choice policies and round availability.
    slots=defaultdict(list)
    for c,file in [('ring','ring/candidate_grid.csv'),(None,'properties/candidate_grid.csv')]:
        for r in rows(D/file):slots[r['slot_id']].append(dict(r,condition=c or r['property']))
    assert len(slots)==1500
    recalculated={};patterns={}
    for sid,items in slots.items():
        assert len(items)==6 and len({(r['turn'],r['candidate_position']) for r in items})==6
        items=sorted(items,key=lambda r:(int(r['turn']),int(r['candidate_position'])))
        eligible=[r for r in items if boolean(r['eligible'])];unique=list({r['canonical']:r for r in eligible}.values())
        first=any(int(r['turn'])==4 for r in eligible);second=any(int(r['turn'])==5 for r in eligible)
        patterns[sid]=(first,second)
        for policy in ['first_eligible','random_unique_expected','best_QED','round1_best_QED','round2_best_QED']:
            selected=eligible if policy=='first_eligible' else unique
            if policy.startswith('round'):selected=[r for r in eligible if int(r['turn'])==(4 if policy=='round1_best_QED' else 5)]
            q=[float(r['QED']) for r in selected]
            u=0 if not q else (q[0] if policy=='first_eligible' else (mean(q) if policy=='random_unique_expected' else max(q)))
            recalculated[(sid,policy)]=(int(bool(q)),u)
    replay=rows(D/'selection_replay7500.csv');assert len(replay)==7500
    for r in replay:
        s,u=recalculated[(r['slot_id'],r['policy'])];equal(r['S'],s);equal(r['U'],u)
        if r['policy']=='best_QED':equal(r['U'],final[r['slot_id']]['U'])
    for file,ring in [('properties/policy_replay.csv',False),('ring/selection_replay.csv',True)]:
        for r in rows(D/file):
            if int(r['k'])!=6 and ring:continue
            p={'uniform_unique_expected':'random_unique_expected','max_QED':'best_QED'}.get(r['policy'],r['policy'])
            s,u=recalculated[(r['slot_id'],p)];equal(r['S'],s);equal(r['U'],u)
    for r in rows(D/'selection_policy_summary60.csv'):
        part=[o for o in replay if o['condition']==r['condition'] and o['group']==r['group'] and o['policy']==r['policy']]
        assert len(part)==125;equal(r['S'],mean(o['S'] for o in part));equal(r['U'],mean(o['U'] for o in part))
    av=rows(D/'round_availability1500.csv');continuous=[r for r in av if r['condition']!='ring']
    assert len(continuous)==1000
    assert sum(patterns[r['slot_id']][0] and not patterns[r['slot_id']][1] for r in continuous)==46
    assert sum(patterns[r['slot_id']][1] for r in continuous)==924
    assert sum(any(patterns[r['slot_id']]) for r in continuous)==970
    for r in rows(D/'round_retention_summary12.csv'):
        part=[v for sid,v in patterns.items() if final[sid]['group']==r['group'] and (('ring' if not sid.startswith(('logp_','tpsa_')) else sid.split('_')[0])==r['condition'])]
        for name,cond in [('both',lambda a,b:a and b),('first_only',lambda a,b:a and not b),('second_only',lambda a,b:b and not a),('neither',lambda a,b:not a and not b)]:equal(r[name],sum(cond(*v) for v in part))
    result=dict(status='passed',docking_observations=759,scored=sum(int(r['docking_ok']) for r in obs),planned_generated=660,
        docking_target_means=231,table3_rows=12,figure6_task_points=132,figure6_means_intervals=21,ring_job_scores_checked=198,
        original_final_output_identities_verified=True,paired_to_same11_targets=True,selection_trajectories=1500,selection_policies_replayed=7500,
        continuous_retention=dict(planned=1000,second_only_selector_delivered=924,union_delivered=970,first_only=46),
        max_tolerance=2e-11,new_API_calls=0,new_docking_jobs=0)
    (D/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
