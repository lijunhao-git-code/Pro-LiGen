"""Join cached MCS and docking with saved final and proposal records. No API calls."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from common import ROOT,read,read_jsonl,sha,now
from analysis_utils import csv,save_json,mcs_key,choose_independent,describe


def structural(properties_dir,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    tasks={r['chemblid']:r for r in read(ROOT/'inputs/tasks25.json')}
    grid=pd.read_csv(properties_dir/'candidate_grid.csv').replace({np.nan:None})
    ff=pd.read_csv(properties_dir/'final_positions.csv').replace({np.nan:None})
    bf=pd.read_csv(properties_dir/'direct_baselines150.csv').replace({np.nan:None})
    cache={}
    for p in sorted((ROOT/'analysis/mcs_cache').glob('*.json')):
        record=read(p);cache[record['key']]=record
    metric_names=['mcs_seed_fraction','mcs_lower_seed_fraction','mcs_upper_seed_fraction']
    feature_rows=[]
    for key,row in cache.items():
        feature_rows.append({k:row[k] for k in ['key','seed','canonical','complete','common_atoms','seed_atoms','candidate_atoms',*metric_names,'elapsed_seconds']})
    csv(outdir/'mcs_unique_pair_features.csv',feature_rows)
    def enrich(records):
        out=[]
        for row in records:
            if row.get('canonical') and (row.get('eligible',True)):
                key=mcs_key(tasks[row['chemblid']]['seed_smiles'],row['canonical']);assert key in cache,key
                row.update({k:cache[key][k] for k in metric_names});row['mcs_complete']=cache[key]['complete']
            else:row.update({k:None for k in metric_names});row['mcs_complete']=None
            out.append(row)
        return out
    candidates=enrich(grid.to_dict('records'));final=enrich(ff.to_dict('records'));baseline=enrich(bf.to_dict('records'))
    csv(outdir/'candidate_grid_with_MCS.csv',candidates);csv(outdir/'final_positions_with_MCS.csv',final);csv(outdir/'direct_baselines_with_MCS.csv',baseline)
    combined=pd.DataFrame(final+baseline)
    metrics=['S','U','QED','SA_raw','tanimoto',*metric_names,'delta_MW','delta_HeavyAtoms','delta_RC']
    target=combined.groupby(['property','chemblid','group'])[metrics].mean().reset_index()
    summary=target.groupby(['property','group'])[metrics].mean().reset_index()
    completion=combined.groupby(['property','group']).agg(planned=('slot_id','size'),delivered=('S','sum'),MCS_complete=('mcs_complete','sum')).reset_index()
    rename={'mcs_seed_fraction':'mcs_completed_cases_mean'}
    csv(outdir/'structural_target_means.csv',target.rename(columns=rename));csv(outdir/'structural_summary.csv',summary.merge(completion).rename(columns=rename))
    byslot={}
    for row in candidates:byslot.setdefault(row['slot_id'],[]).append(row)
    curve=[]
    for original in final:
        pool=byslot[original['slot_id']]
        for metric,field,bound in [('tanimoto','tanimoto','exact'),('mcs_seed_coverage','mcs_lower_seed_fraction','guaranteed'),('mcs_seed_coverage','mcs_upper_seed_fraction','possible')]:
            for threshold in np.arange(10)/10:
                filtered=[r for r in pool if r['eligible'] and r[field]>=threshold-1e-12]
                reselected=choose_independent(filtered)
                final_pass=original['canonical'] is not None and original.get(field) is not None and original[field]>=threshold-1e-12
                for policy,support in [('filter_original_final',original if final_pass else None),('reselect_saved_eligible',reselected)]:
                    row={k:original[k] for k in ['slot_id','property','chemblid','group','position']}
                    row.update(metric=metric,bound=bound,threshold=float(threshold),policy=policy,S=int(support is not None),
                        U=support['QED'] if support is not None else 0.,QED=support['QED'] if support is not None else None,
                        tanimoto=support.get('tanimoto') if support is not None else None,selected_canonical=support.get('canonical') if support is not None else None)
                    curve.append(row)
    cf=pd.DataFrame(curve);csv(outdir/'structure_sensitivity_by_slot.csv',cf)
    ids=['property','chemblid','group','metric','bound','threshold','policy']
    ct=cf.groupby(ids)[['S','U','QED','tanimoto']].mean().reset_index();csv(outdir/'structure_sensitivity_target_means.csv',ct)
    cs=ct.groupby([c for c in ids if c!='chemblid'])[['S','U','QED','tanimoto']].mean().reset_index()
    csv(outdir/'structure_sensitivity_summary.csv',cs)
    zero=cs[(cs.threshold==0)&cs.policy.eq('reselect_saved_eligible')]
    original_summary=ff.groupby(['property','group'])[['S','U']].mean()
    for r in zero.to_dict('records'):
        expect=original_summary.loc[(r['property'],r['group'])]
        assert abs(r['S']-expect.S)<1e-12 and abs(r['U']-expect.U)<1e-12
    report=dict(status='passed',time_utc=now(),mcs_unique_pairs=len(cache),mcs_incomplete=sum(not r['complete'] for r in cache.values()),
        final_positions=len(final),direct_baselines=len(baseline),candidate_positions=len(candidates),
        all_eligible_candidates_and_final_outputs_have_cached_MCS=True,
        MCS_bounds='Exact on completed searches; lower/upper bounds retained on timeouts. Sensitivity curves separately show guaranteed/possible retention.',
        MCS_aggregate_columns='mcs_completed_cases_mean uses completed cases only; lower/upper means include every delivered molecule. Different denominators are explicitly named.',
        thresholds='0.0 to0.9; predeclared descriptive sensitivity, not extra generation or primary hard constraints.',
        zero_threshold_curves_match_primary_S_U=True,new_API_calls=0)
    save_json(outdir/'structural_verification.json',report);print('STRUCTURAL ANALYSIS',report,flush=True)


def docking(properties_dir,outdir):
    outdir.mkdir(parents=True,exist_ok=True)
    obs=pd.read_csv(ROOT/'docking/observations1175.csv').replace({np.nan:None})
    assert len(obs)==1175 and obs.observation_id.is_unique
    assert not obs.docking_status.isin(['pending_generation','pending_docking']).any(),'Docking not finished'
    lookup={}
    for file,kind in [('final_positions.csv','generated'),('direct_baselines150.csv','direct_baseline')]:
        for r in pd.read_csv(properties_dir/file).replace({np.nan:None}).to_dict('records'):lookup[(kind,r['slot_id'])]=r
    seed_scores={r['chemblid']:r['vina'] for r in obs.to_dict('records') if r['kind']=='seed' and r['docking_status']=='ok'}
    records=[]
    for row in obs.to_dict('records'):
        if row['kind']=='seed':continue
        sid=row['observation_id'].removeprefix('generated_').removeprefix('direct_')
        source=lookup[(row['kind'],sid)]
        assert (row['canonical'] or None)==(source['canonical'] or None)
        if row['job_id']:
            job=read(ROOT/'docking/jobs'/row['job_id']/'result.json')
            assert job['canonical']==row['canonical'] and job['chemblid']==row['chemblid']
            if row['docking_status']=='ok':assert abs(job['vina']-row['vina'])<1e-12
        row.update(slot_id=sid,QED=source['QED'],SA_raw=source['SA_raw'],S=source['S'],U=source['U'],tanimoto=source['tanimoto'],
            delta_MW=source['delta_MW'],delta_HeavyAtoms=source['delta_HeavyAtoms'],seed_vina=seed_scores.get(row['chemblid']),
            docking_ok=int(row['docking_status']=='ok'),
            delta_vina=row['vina']-seed_scores[row['chemblid']] if row['docking_status']=='ok' and row['chemblid'] in seed_scores else None)
        records.append(row)
    df=pd.DataFrame(records);csv(outdir/'docking_all1150_output_records.csv',df)
    sub=df[df.in_primary_docking_subset]
    assert len(sub[sub.kind.eq('generated')])==440 and len(sub[sub.kind.eq('direct_baseline')])==66
    csv(outdir/'docking_primary506_output_records.csv',sub)
    target=sub.groupby(['property','kind','group','chemblid'])[['S','U','docking_ok','vina','delta_vina','QED','tanimoto','delta_MW']].mean().reset_index()
    csv(outdir/'docking_target_means.csv',target)
    summary=target.groupby(['property','kind','group'])[['S','U','docking_ok','vina','delta_vina','QED','tanimoto','delta_MW']].mean().reset_index()
    counts=sub.groupby(['property','kind','group']).agg(planned_output_records=('slot_id','size'),eligible_deliveries=('S','sum'),
        docking_completed=('docking_ok','sum'),unique_jobs=('job_id','nunique')).reset_index()
    csv(outdir/'docking_summary.csv',summary.merge(counts))
    csv(outdir/'docking_status_counts.csv',sub.groupby(['property','kind','group','docking_status']).size().rename('count').reset_index())
    seedframe=obs[obs.kind.eq('seed')&obs.in_primary_docking_subset];csv(outdir/'seed_docking11.csv',seedframe)
    report=dict(status='passed',time_utc=now(),all_observations=len(obs),generated_primary_records=440,direct_primary_records=66,
        seed_primary_records=len(seedframe),seed_successful=len(seed_scores),
        all_unique_job_results_match_observation_scores_and_structures=True,
        failure_scores_remain_missing=True,delta_vina='Output minus same-target seed; positive is less favorable.',
        target_weighting='Conditional valid scores within target, then equal means across available targets; planned/completed counts retained.',
        new_API_calls=0)
    save_json(outdir/'docking_verification.json',report);print('DOCKING ANALYSIS',report,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group();mode.add_argument('--docking-only',action='store_true');mode.add_argument('--structure-only',action='store_true');p.add_argument('--output-base');args=p.parse_args()
    base=ROOT/(args.output_base or 'analysis');properties_dir=base/'properties'
    if not args.docking_only:structural(properties_dir,base/'structures')
    if not args.structure_only:docking(properties_dir,base/'docking')
