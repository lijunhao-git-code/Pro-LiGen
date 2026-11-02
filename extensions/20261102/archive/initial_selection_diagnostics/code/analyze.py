"""Offline candidate quality and selection-policy audit. No network or generation."""
from collections import defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import Descriptors,QED,rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer
from retained_methods import choose as old_choose,interval,paired,correct_holm

ROOT=Path(__file__).resolve().parents[1]
GROUPS=['T','M','G']
POLICIES=['first_eligible','random_unique_expected','best_QED','random_positions_expected','random_unique_draw42']
METRICS=['delivery','yield_QED','QED','SA_raw','tanimoto','dHA_organic','dMW_organic',
         'mcs_seed_fraction','mcs_lower_seed_fraction','mcs_upper_seed_fraction']
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
RDLogger.DisableLog('rdApp.*')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(n):return json.loads((ROOT/'inputs'/n).read_text(encoding='utf-8'))
def csv(name,rows):pd.DataFrame(rows).to_csv(ROOT/'results'/name,index=False,encoding='utf-8-sig')
def dump(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


@lru_cache(maxsize=None)
def describe(s):
    m=Chem.MolFromSmiles(s)
    if m is None or m.GetNumHeavyAtoms()==0:return None
    return dict(canonical=Chem.MolToSmiles(m),QED=float(QED.qed(m)),
                SA_raw=float(sascorer.calculateScore(m)),MW=float(Descriptors.MolWt(m)),
                RC=int(Descriptors.RingCount(m)),HeavyAtoms=m.GetNumHeavyAtoms(),
                fragments=len(Chem.GetMolFrags(m)),fp=FP.GetFingerprint(m))


def eligible_unique(rows):
    return sorted({r['canonical']:r for r in rows if r['eligible']}.values(),key=lambda r:r['canonical'])


def select(rows,policy,slot_id):
    valid=[r for r in rows if r['eligible']]
    unique=eligible_unique(rows)
    if not valid:return [],None
    if policy=='first_eligible':return [valid[0]],valid[0]
    if policy=='best_QED':
        row=min(unique,key=lambda r:(-r['QED'],r['SA_raw'],r['canonical']))
        return [row],row
    if policy=='random_unique_expected':return unique,None
    if policy=='random_positions_expected':return valid,None
    if policy=='random_unique_draw42':
        seed=int.from_bytes(hashlib.sha256(('42|'+slot_id).encode()).digest()[:16],'big')
        row=unique[int(np.random.default_rng(seed).integers(len(unique)))]
        return [row],row
    raise ValueError(policy)


def policy_row(entry,rows,policy,k):
    support,chosen=select(rows,policy,entry['slot_id'])
    row=dict(slot_id=entry['slot_id'],chemblid=entry['chemblid'],group=entry['group'],
             run_position=entry['position'],policy=policy,k=k,delivery=int(bool(support)),
             selected_canonical=chosen['canonical'] if chosen else None,
             selection_kind='analytical_expectation' if 'expected' in policy else 'one_selected_structure',
             eligible_positions=sum(r['eligible'] for r in rows),eligible_unique=len(eligible_unique(rows)),
             random_support_n=len(support) if 'expected' in policy else None,
             yield_QED=float(np.mean([r['QED'] for r in support])) if support else 0.)
    for key in METRICS[2:]:
        values=[r.get(key) for r in support]
        row[key]=float(np.mean(values)) if values and all(v is not None and np.isfinite(v) for v in values) else None
    return row


def main():
    started=time.monotonic()
    (ROOT/'results').mkdir(exist_ok=True)
    if (ROOT/'results/verification.json').exists():raise RuntimeError('Use a fresh directory to rerun')
    frozen=json.loads((ROOT/'frozen_inputs_and_code.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==v for p,v in frozen.items())
    original=read('B_planned375.json');candidates=read('B_candidates_lossless2244.json')
    tasks={t['chemblid']:t for t in read('tasks25.json')}
    refs=defaultdict(set)
    for r in read('B_references150.json'):refs[(r['chemblid'],r['group'])].add(r['canonical'])
    features=pd.read_csv(ROOT/'inputs/P045_molecule_structure_features.csv')
    fmap={(r['chemblid'],r['canonical']):r for r in features.to_dict('records')}
    originals={r['slot_id']:r for r in original}
    assert len(originals)==375 and len(tasks)==25 and len(candidates)==2244
    raw_by_slot=defaultdict(list)
    for c in candidates:raw_by_slot[c['slot_id']].append(c)
    grid=[];by_slot={};saved_rows=0
    for entry in original:
        sid=entry['slot_id'];items=raw_by_slot[sid]
        source={(r['turn'],r['candidate_position']):r for r in items}
        assert len(source)==len(items)
        rows=[];seed=describe(tasks[entry['chemblid']]['seed_smiles'])
        for index in range(1,7):
            turn=4+(index-1)//3;pos=1+(index-1)%3
            original_row=source.get((turn,pos))
            row=dict(original_row) if original_row else dict(turn=turn,candidate_position=pos,
                valid=False,eligible=False,canonical=None,smiles=None,reason='no_retained_completed_attempt_position')
            row.update(slot_id=sid,chemblid=entry['chemblid'],group=entry['group'],
                       run_position=entry['position'],proposal_index=index,archived=original_row is not None)
            saved_rows+=int(row['archived'])
            if row['valid']:
                d=describe(row['smiles']);assert d is not None and d['canonical']==row['canonical']
                for key in ['QED','SA_raw','MW','RC','HeavyAtoms','fragments']:
                    assert abs(d[key]-row[key])<1e-9,(sid,key)
                expected=(d['RC']==tasks[row['chemblid']]['target_rc'] and d['fragments']==1
                          and d['canonical']!=seed['canonical'] and d['canonical'] not in refs[(row['chemblid'],row['group'])])
                assert row['eligible']==expected
                f=fmap[(row['chemblid'],row['canonical'])]
                row['tanimoto']=float(DataStructs.TanimotoSimilarity(seed['fp'],d['fp']))
                assert abs(row['tanimoto']-f['tanimoto_parent'])<1e-12
                for key in ['dHA_organic','dMW_organic','mcs_seed_fraction','mcs_lower_seed_fraction','mcs_upper_seed_fraction']:
                    row[key]=f[key] if pd.notna(f.get(key)) else None
                row['QED_above_seed']=d['QED']>seed['QED']
            else:
                assert not row['eligible']
                if row.get('smiles'):assert describe(row['smiles']) is None
            row['valid_QED_utility']=row['QED'] if row['valid'] else 0.
            row['eligible_QED_utility']=row['QED'] if row['eligible'] else 0.
            rows.append(row);grid.append(row)
        by_slot[sid]=rows
    assert saved_rows==2244 and len(grid)==2250
    print('All 2,244 archived positions and all valid chemical descriptors verified; six missing positions retained as failures.',flush=True)
    # Reproduce every stored position/policy identity, not just rounded summary tables.
    old=read('B_fixed_candidate_replay1500.json')
    for record in old:
        chosen=old_choose(raw_by_slot[record['slot_id']],record['policy'])
        assert record['selected_canonical']==(chosen['canonical'] if chosen else None)
        assert abs(record['compliant_QED_utility']-(chosen['QED'] if chosen else 0))<1e-12
    assert len(old)==1500
    policies=[];support=[]
    for entry in original:
        rows=by_slot[entry['slot_id']]
        for k in range(1,7):
            prefix=rows[:k]
            out=[policy_row(entry,prefix,policy,k) for policy in POLICIES]
            assert len(set(r['delivery'] for r in out))==1
            assert out[2]['yield_QED']+1e-12>=max(r['yield_QED'] for r in out)
            policies.extend(out)
            opts=eligible_unique(prefix)
            support.append(dict(slot_id=entry['slot_id'],k=k,canonical=[r['canonical'] for r in opts],
                                QED=[r['QED'] for r in opts],probability_each=1/len(opts) if opts else None))
        for turn,label in [(4,'round1_best_QED'),(5,'round2_best_QED')]:
            rows_turn=[r for r in rows if r['turn']==turn]
            record=policy_row(entry,rows_turn,'best_QED',3)
            record['policy']=label;policies.append(record)
        deployed=next(r for r in policies[-32:] if r['slot_id']==entry['slot_id'] and r['k']==6 and r['policy']=='best_QED')
        assert deployed['selected_canonical']==entry['canonical']
        assert abs(deployed['yield_QED']-entry['compliant_QED_utility'])<1e-12
        first=[r for r in policies[-32:] if r['slot_id']==entry['slot_id'] and r['policy']=='first_eligible']
        best=[r for r in policies[-32:] if r['slot_id']==entry['slot_id'] and r['policy']=='best_QED']
        assert (np.diff([r['yield_QED'] for r in first])>=-1e-12).all()
        assert (np.diff([r['yield_QED'] for r in best])>=-1e-12).all()
    pf=pd.DataFrame(policies)
    target=pf.groupby(['group','chemblid','policy','k'])[METRICS].mean().reset_index()
    summaries=target.groupby(['group','policy','k'])[METRICS].mean().reset_index()
    count=pf.groupby(['group','policy','k']).agg(planned=('slot_id','size'),delivered=('delivery','sum')).reset_index()
    summaries=summaries.merge(count,on=['group','policy','k'],validate='one_to_one')
    assert count.planned.eq(125).all()
    profiles=[];slot_profiles=[]
    gf=pd.DataFrame(grid)
    for entry in original:
        part=gf[gf.slot_id==entry['slot_id']]
        eligible=part[part.eligible]
        slot_profiles.append(dict(slot_id=entry['slot_id'],chemblid=entry['chemblid'],group=entry['group'],
            archived_n=int(part.archived.sum()),valid_n=int(part.valid.sum()),eligible_n=int(part.eligible.sum()),
            eligible_unique_n=eligible.canonical.nunique(),eligible_QED_min=float(eligible.QED.min()) if len(eligible) else None,
            eligible_QED_max=float(eligible.QED.max()) if len(eligible) else None,
            eligible_QED_mean_positions=float(eligible.QED.mean()) if len(eligible) else None))
    for (group,task),part in gf.groupby(['group','chemblid']):
        valid=part[part.valid];eligible=part[part.eligible]
        assert len(part)==30
        profiles.append(dict(group=group,chemblid=task,planned_positions=30,archived=int(part.archived.sum()),
            valid=int(part.valid.sum()),eligible=int(part.eligible.sum()),valid_fraction=float(part.valid.mean()),
            eligible_fraction=float(part.eligible.mean()),valid_proposal_QED_yield=float(part.valid_QED_utility.mean()),
            eligible_proposal_QED_yield=float(part.eligible_QED_utility.mean()),
            valid_QED=float(valid.QED.mean()),eligible_QED=float(eligible.QED.mean()),
            valid_QED_median=float(valid.QED.median()),eligible_QED_median=float(eligible.QED.median()),
            valid_QED_above_seed=float(valid.QED_above_seed.mean()),
            eligible_QED_above_seed=float(eligible.QED_above_seed.mean())))
    candidate_targets=pd.DataFrame(profiles)
    candidate_summary=candidate_targets.groupby('group').mean(numeric_only=True).reset_index()
    totals=candidate_targets.groupby('group')[['planned_positions','archived','valid','eligible']].sum().reset_index()
    for col in ['planned_positions','archived','valid','eligible']:candidate_summary[col+'_total']=totals.set_index('group').loc[candidate_summary.group,col].to_numpy()
    effects=[]
    for group in GROUPS:
        part=target[(target.group==group)&(target.k==6)&target.policy.isin(['first_eligible','random_unique_expected','best_QED'])]
        matrix=part.pivot(index='chemblid',columns='policy',values='yield_QED')
        for base in ['first_eligible','random_unique_expected']:
            effects.append(dict(group=group,comparison='best_QED-'+base,n_targets=25,
                                interpretation='conditional saved-pool selection contrast, no new generation',
                                **interval((matrix.best_QED-matrix[base]).to_numpy())))
    tests=[]
    for outcome in ['random_unique_expected_yield','valid_proposal_QED_yield']:
        if outcome=='random_unique_expected_yield':
            matrix=target[(target.k==6)&(target.policy=='random_unique_expected')].pivot(index='chemblid',columns='group',values='yield_QED')
        else:matrix=candidate_targets.pivot(index='chemblid',columns='group',values=outcome)
        for left,right in [('M','T'),('G','T'),('G','M')]:
            tests.append(dict(outcome=outcome,comparison=left+'-'+right,
                family='six_exploratory_candidate_and_random_selection_tests',**paired((matrix[left]-matrix[right]).to_numpy())))
    correct_holm(tests)
    # Descriptive turn-wise proposal quality; round 2 depends on full round-1 feedback.
    turn_rows=[]
    for (group,task,turn),part in gf.groupby(['group','chemblid','turn']):
        turn_rows.append(dict(group=group,chemblid=task,turn=int(turn),planned=len(part),
            valid_fraction=float(part.valid.mean()),eligible_fraction=float(part.eligible.mean()),
            valid_QED_yield=float(part.valid_QED_utility.mean()),
            valid_QED=float(part.loc[part.valid,'QED'].mean()),eligible_QED=float(part.loc[part.eligible,'QED'].mean())))
    csv('candidate_grid2250.csv',grid);csv('selection_by_slot12000.csv',policies)
    csv('policy_target_means.csv',target);csv('policy_summary.csv',summaries)
    csv('full_pool_policy_summary.csv',summaries[(summaries.k==6)|summaries.policy.str.startswith('round')])
    csv('candidate_target_quality75.csv',profiles);csv('candidate_quality_summary3.csv',candidate_summary)
    csv('slot_candidate_counts375.csv',slot_profiles);csv('selection_effects6.csv',effects)
    csv('group_comparisons6.csv',tests);csv('turn_target_quality150.csv',turn_rows)
    with (ROOT/'results/random_unique_support2250.jsonl').open('w',encoding='utf-8') as stream:
        for row in support:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    dump('results/verification.json',dict(status='passed',all_frozen_files_unchanged=all(sha(ROOT/p)==v for p,v in frozen.items()),
        original_375_selected_identities_exact=True,old_1500_replay_records_exact=True,
        archived_candidate_rows=2244,planned_candidate_grid_rows=2250,missing_archived_positions=6,
        all_valid_descriptors_and_eligibility_recomputed=True,
        original_parent_Tanimoto_recomputed=True,policies_have_identical_delivery_for_each_prefix=True,
        max_QED_ge_all_alternative_policy_expectations=True,first_and_best_prefix_yields_monotone=True,
        selection_rows=len(policies),random_support_rows=len(support),
        new_test_family='6 exploratory paired target tests, Holm over all six',n_targets=25,
        random_seed=42,random_primary='exact expectation over distinct eligible canonical structures',
        saved_pools_not_independent_generation_experiments=True,
        new_API_calls=0,new_training_jobs=0,new_docking_jobs=0,
        elapsed_seconds=round(time.monotonic()-started,3),
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,rdkit=rdBase.rdkitVersion)))
    print('Numerical audit complete. Full saved-pool policy results:',flush=True)
    print(summaries[summaries.k.eq(6)][['group','policy','delivered','yield_QED','QED','tanimoto']].to_string(index=False),flush=True)
    print(pd.DataFrame(tests).to_string(index=False),flush=True)


if __name__=='__main__':main()
