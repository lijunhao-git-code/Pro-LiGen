"""Derived B02 docking summaries and independent checks; no model/API calls."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
assert ROOT.name.startswith('neurocomputing-b02-') and ROOT.parent==Path('/tmp')


def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def mean_ci(values):
    values=np.array(values,dtype=float)
    if not len(values):
        return None,None,None
    rng=np.random.default_rng(42)
    boot=values[rng.integers(0,len(values),size=(10000,len(values)))].mean(axis=1)
    ci=np.quantile(boot,[.025,.975])
    return float(values.mean()),float(ci[0]),float(ci[1])


def calibration():
    results=json.loads((ROOT/'calibration_results.json').read_text())
    audit=json.loads((ROOT/'receptor_audit.json').read_text())
    mapping={r['chemblid']:r for r in results}
    rows=[]
    for r in audit:
        x=mapping.get(r['chemblid'],{})
        primary=bool(x.get('calibration_passed',False) and r.get('primary_nonmetal',False) and x.get('status')=='ok')
        rows.append(dict(chemblid=r['chemblid'],uniprotid=r['uniprotid'],pdb=r.get('pdb_id'),
             ligand=r.get('cocrystal_ligand'),chains=','.join(r.get('chains',[])),source_url=r.get('source_url'),
             prep_status=r['status'],redock_status=x.get('status','not_run'),metal_atoms=r.get('metal_atom_count'),
             resolution_A=r.get('resolution_A'),reference_heavy_atoms=r.get('reference_heavy_atoms'),
             top1_rmsd_A=x.get('top1_rmsd_A'),best_top10_rmsd_A=x.get('top10_best_rmsd_A'),
             top1_pose_calibration_pass=x.get('calibration_passed',False),primary_nonmetal_calibrated=primary,
             mmff_status=x.get('mmff_status'),reason=r.get('reason') or x.get('reason'),
             caveat=r.get('pocket_seed_caveat','No seed binding-mode validation')))
    frame=pd.DataFrame(rows)
    assert len(frame)==25 and frame.chemblid.nunique()==25
    frame.to_csv(ROOT/'receptor_calibration_summary.csv',index=False)
    dump(ROOT/'calibration_summary.json',dict(targets=25,prepared=len(results),redock_successful=sum(x['status']=='ok' for x in results),
         top1_pose_passed=int(frame.top1_pose_calibration_pass.sum()),
         primary_nonmetal_calibrated=int(frame.primary_nonmetal_calibrated.sum()),
         primary_targets=frame.loc[frame.primary_nonmetal_calibrated,'chemblid'].tolist(),
         seed_binding_mode_verified=False,not_affinity_validation=True,
         top1_cutoff_A=2.0,no_ligand_superposition=True,no_generated_score_based_target_selection=True))


def scores():
    frame=pd.read_csv(ROOT/'docking_scores.csv')
    assert len(frame)==900 and not frame.observation_id.duplicated().any()
    assert len(frame.query("study=='A'"))==500 and len(frame.query("study=='B'"))==375
    assert len(frame.query("kind=='seed'"))==25
    assert frame.loc[frame.docking_status!='ok','vina'].isna().all()
    assert np.isfinite(frame.loc[frame.docking_status=='ok','vina']).all()
    results=json.loads((ROOT/'docking_results.json').read_text())
    tasks=json.loads((ROOT/'docking_tasks.json').read_text())
    mapping={r['job_id']:r for r in results}
    assert len(results)==len(tasks)==len(mapping)
    assert set(frame.job_id.dropna())==set(mapping)
    for r in frame.itertuples():
        if pd.notna(r.job_id):
            assert mapping[r.job_id]['chemblid']==r.chemblid
            if r.docking_status=='ok':
                assert r.vina==mapping[r.job_id]['vina']
                assert r.active_component==mapping[r.job_id]['canonical']
    seeds=frame.query("kind=='seed' and docking_status=='ok'").set_index('chemblid')['vina']
    frame['seed_vina_same_protocol']=frame.chemblid.map(seeds)
    frame['dVina_output_minus_seed']=frame.vina-frame.seed_vina_same_protocol
    frame['seed_score_improved']=np.where(frame.dVina_output_minus_seed.notna(),
                                        (frame.dVina_output_minus_seed<0).astype(float),np.nan)
    frame['dQED_active_output_minus_seed']=frame.active_QED-frame.chemblid.map(frame.query("kind=='seed'").set_index('chemblid').active_QED)
    frame.to_csv(ROOT/'docking_scores_with_seed_pairs.csv',index=False)
    frame.groupby(['study','group','docking_status']).size().rename('record_count').reset_index().to_csv(ROOT/'docking_coverage.csv',index=False)
    changed=frame.loc[frame.parent_fragment_removed==True]
    changed.to_csv(ROOT/'salt_parent_active_sensitivity.csv',index=False)
    expected=json.loads((ROOT/'protected_before.json').read_text())
    changed_input=[p for p,h in expected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed_input
    group_rows,target_rows=[] ,[]
    common_lists={}
    for study,groups in [('A',['NG','TG','PG-aggr','PG-mild']),('B',['T','M','G'])]:
        df=frame.query('study==@study')
        evaluated=df.query("docking_status=='ok' and seed_vina_same_protocol.notna()")
        target_groups=evaluated.groupby('chemblid').group.nunique()
        common=sorted(target_groups.index[target_groups==len(groups)])
        common_lists[study]=common
        for group in groups:
            g=df.query('group==@group')
            valid=g.query("docking_status=='ok'")
            pair=valid[valid.chemblid.isin(common)]
            means=pair.groupby('chemblid')[['vina','dVina_output_minus_seed','seed_score_improved','active_QED','active_SA_raw','dQED_active_output_minus_seed']].mean()
            for t,r in means.iterrows():
                target_rows.append(dict(study=study,group=group,chemblid=t,**r.to_dict(),
                                   output_success_n=int((pair.chemblid==t).sum())))
            row=dict(study=study,group=group,planned_or_retained_positions=len(g),
                     successful_docking_records=len(valid),successful_docking_targets=valid.chemblid.nunique(),
                     paired_common_target_n=len(common),paired_common_output_n=len(pair),
                     failed_with_selected_receptor=int((g.docking_status.isin(['timeout','technical_failure','worker_fail'])).sum()),
                     excluded_docking_only=int((g.docking_status=='excluded_receptor_or_calibration').sum()),
                     no_generated_output=int((g.docking_status=='no_generated_output').sum()),
                     positive_vina_retained=int((valid.vina>0).sum()))
            for metric in means.columns:
                mean,lo,hi=mean_ci(means[metric])
                row.update({metric+'_target_mean':mean,metric+'_CI95_low':lo,metric+'_CI95_high':hi})
            row['inference']='exploratory; unadjusted descriptive intervals; no bioactivity/equivalence claim'
            group_rows.append(row)
    pd.DataFrame(group_rows).to_csv(ROOT/'docking_group_summary.csv',index=False)
    pd.DataFrame(target_rows).to_csv(ROOT/'docking_target_means.csv',index=False)
    # Show chemical-form sensitivity on ALL records, not only docking-selected subset.
    sensitivity=[]
    for (study,group),g in frame.query("kind=='generated' and parent_smiles.notna()").groupby(['study','group']):
        parent=g.groupby('chemblid')[['parent_QED','parent_SA_raw','parent_MW']].mean().mean()
        active=g.groupby('chemblid')[['active_QED','active_SA_raw','active_MW']].mean().mean()
        sensitivity.append(dict(study=study,group=group,n_records=len(g),n_targets=g.chemblid.nunique(),
             parent_QED=parent.parent_QED,active_QED=active.active_QED,
             parent_SA_raw=parent.parent_SA_raw,active_SA_raw=active.active_SA_raw,
             parent_MW=parent.parent_MW,active_MW=active.active_MW,
             salt_record_n=int((g.parent_fragment_removed==True).sum()),primary_B_selection_changed=False))
    pd.DataFrame(sensitivity).to_csv(ROOT/'chemical_form_group_sensitivity.csv',index=False)
    dump(ROOT/'independent_docking_verification.json',dict(observations900=True,A500_unchanged=True,B375_including_failures=True,
         seeds25=True,unique_job_to_observation_mapping_checked=True,
         input_integrity_checked=len(expected),changed_original_inputs=[],
         successful_jobs=sum(r['status']=='ok' for r in results),unique_jobs=len(results),
         positive_vina_observations=int((frame.vina>0).sum()),missing_scores_not_zero=True,
         common_paired_targets=common_lists,salt_parent_and_active_both_retained=True,
         no_new_molecule_generation_in_docking=True,old_vina_not_used=True))


def rmsd_fixture():
    from rdkit import Chem
    from rdkit.Chem import AllChem,rdMolAlign
    from b02_docking import active
    m=Chem.AddHs(Chem.MolFromSmiles('CCO'))
    assert AllChem.EmbedMolecule(m,randomSeed=42)==0
    a=Chem.RemoveHs(m)
    b=Chem.Mol(a)
    conf=b.GetConformer()
    for i in range(b.GetNumAtoms()):
        pt=conf.GetAtomPosition(i)
        conf.SetAtomPosition(i,(pt.x+5,pt.y,pt.z))
    assert abs(rdMolAlign.CalcRMS(a,b)-5)<1e-8
    assert rdMolAlign.GetBestRMS(Chem.Mol(a),b)<1e-6
    assert active('C[N+](C)(C)C.[Br-]')=='C[N+](C)(C)C'
    try:
        active('CCO.CCN')
        raise AssertionError('ambiguous multi-organic fragments accepted')
    except ValueError:
        pass
    print('RMSD NO-ALIGNMENT, SYMMETRY, SALT/AMBIGUITY TESTS PASSED',flush=True)


def reuse_b():
    """Reuse only unchanged B protocol/receptor/actual prepared ligands, not legacy A."""
    from b02_docking import active
    b=Path('/root/PG-TALIRAGEN/PG-TALIRAGEN/mw_experiments/gin_downstream_20261011_375')
    oldp=json.loads((b/'protocol.json').read_text())
    newp=json.loads((ROOT/'docking_protocol.json').read_text())
    assert oldp['docking']['seed']==newp['seed']==42
    assert oldp['docking']['exhaustiveness']==newp['exhaustiveness']==16
    assert oldp['docking']['n_poses']==newp['n_poses']==10
    assert oldp['docking']['cpu_per_job']==newp['cpu']==1
    assert oldp['docking']['sf_name']==newp['sf_name']=='vina'
    assert oldp['docking']['timeout_seconds']==newp['timeout_seconds']==600
    oldlock=json.loads((b/'locked.json').read_text())
    for file in ('run_docking.py','common.py','protocol.json'):
        assert hashlib.sha256((b/file).read_bytes()).hexdigest()==oldlock['files'][file]
    oldtasks=json.loads((b/'docking/tasks.json').read_text())
    index={(r['chemblid'],active(r['canonical'])):r for r in oldtasks}
    oldrecs=json.loads((b/'inputs/receptors.json').read_text())
    tasks=json.loads((ROOT/'docking_tasks.json').read_text())
    proof=[]
    for task in tasks:
        d=ROOT/'docking'/task['job_id']
        assert not (d/'result.json').exists(), 'Reuse phase must precede new docking'
        old=index.get((task['chemblid'],task['canonical']))
        if not old:
            continue
        rec=oldrecs[task['chemblid']]
        if rec['pdbqt_sha256']!=task['receptor_sha256'] or old['box_center']!=task['box_center'] or old['box_size']!=task['box_size']:
            continue
        source=b/'docking/jobs'/old['job_id']
        result_path=source/'result.json'
        if not result_path.exists():
            continue
        r=json.loads(result_path.read_text())
        if r['status']!='ok' or r.get('docked_fragment')!=task['canonical']:
            continue
        assert np.isfinite(r['vina']) and r['vina']==r['energies'][0][0]
        d.mkdir(parents=True,exist_ok=True)
        for name in ('ligand.pdbqt','poses.pdbqt'):
            assert (source/name).exists()
            shutil.copy2(source/name,d/name)
        digest=hashlib.sha256(result_path.read_bytes()).hexdigest()
        new=dict(task,status='ok',vina=r['vina'],energies=r['energies'],mmff_status=r.get('mmff_optimize_status'),
                 reused_unchanged_B_calculation=True,reuse_source_result=str(result_path),reuse_source_sha256=digest,
                 original_started_utc=r.get('started_utc'),original_finished_utc=r.get('finished_utc'))
        dump(d/'result.json',new)
        proof.append(dict(job_id=task['job_id'],target=task['chemblid'],source_result=str(result_path),
                     source_sha256=digest,source_protocol_sha256=hashlib.sha256((b/'protocol.json').read_bytes()).hexdigest(),
                     receptor_sha256=task['receptor_sha256'],prepared_canonical=task['canonical'],same_box=True,
                     seed42_exhaust16_poses10=True,same_RDKIT_MMFF500_Meeko_ligand_pipeline=True))
    dump(ROOT/'reused_B_cache_proof.json',proof)
    print('REUSED UNCHANGED B JOBS',len(proof),'NEW JOBS',len(tasks)-len(proof),flush=True)


def form_sensitivity():
    from b02_docking import active
    from rdkit import Chem
    b=Path('/root/PG-TALIRAGEN/PG-TALIRAGEN/mw_experiments/gin_downstream_20261011_375')
    import csv
    arefs={}
    for r in csv.DictReader((ROOT/'A_historical_reference_properties.csv').open()):
        arefs.setdefault((r['chemblid'],r['group']),set()).add(active(r['canonical']))
    bref={}
    all_reference_fragments={}
    reference_form_records=[]
    unknown_refs={}
    for line in (b/'inputs/prompts.jsonl').read_text().splitlines():
        r=json.loads(line)
        key=(r['chemblid'],r['group'])
        bref[key]=set()
        all_reference_fragments[key]=set()
        unknown_refs[key]=0
        for rank,s in enumerate(r['exemplars'],1):
            all_reference_fragments[key].update(Chem.MolToSmiles(m,isomericSmiles=True)
                for m in Chem.GetMolFrags(Chem.MolFromSmiles(s),asMols=True))
            try:
                component=active(s)
                bref[key].add(component)
                error=None
            except ValueError as exc:
                component=None
                error=str(exc)
                unknown_refs[key]+=1
            reference_form_records.append(dict(chemblid=r['chemblid'],group=r['group'],rank=rank,
                                          parent_smiles=s,active_component=component,reason=error))
    pd.DataFrame(reference_form_records).to_csv(ROOT/'B_reference_active_component_inventory.csv',index=False)
    seeds={r['chemblid']:active(r['seed_canonical']) for r in csv.DictReader((b/'inputs/targets.csv').open())}
    frame=pd.DataFrame(json.loads((ROOT/'observations.json').read_text()))
    rows=[]
    for r in frame.query("kind=='generated' and parent_smiles.notna()").itertuples():
        refs=arefs.get((r.chemblid,r.group),set()) if r.study=='A' else bref[(r.chemblid,r.group)]
        rows.append(dict(observation_id=r.observation_id,study=r.study,group=r.group,chemblid=r.chemblid,
             parent_smiles=r.parent_smiles,active_component=r.active_component,
             component_seed_copy=r.active_component==seeds[r.chemblid],
             component_reference_copy=r.active_component in refs,
             exact_copy_of_any_reference_component=(r.active_component in refs if r.study=='A' else
                                                    r.active_component in all_reference_fragments[(r.chemblid,r.group)]),
             component_reference_noncopy_fully_evaluable=(r.study=='A' or unknown_refs[(r.chemblid,r.group)]==0),
             parent_fragment_removed=r.parent_fragment_removed,
             original_selection_not_changed=True))
    df=pd.DataFrame(rows)
    df.to_csv(ROOT/'active_component_noncopy_sensitivity_all_outputs.csv',index=False)
    df.query('component_seed_copy or component_reference_copy or exact_copy_of_any_reference_component').to_csv(ROOT/'active_component_copy_flags.csv',index=False)
    summary=df.groupby(['study','group']).agg(n_outputs=('observation_id','size'),
         active_seed_copies=('component_seed_copy','sum'),active_reference_copies=('component_reference_copy','sum'),
         parent_fragment_removed=('parent_fragment_removed','sum'),
         exact_copies_of_any_reference_component=('exact_copy_of_any_reference_component','sum'),
         reference_noncopy_fully_evaluable=('component_reference_noncopy_fully_evaluable','sum')).reset_index()
    summary.to_csv(ROOT/'active_component_noncopy_summary.csv',index=False)
    chemical=[]
    for (study,group),g in frame.query("kind=='generated' and parent_smiles.notna()").groupby(['study','group']):
        parent=g.groupby('chemblid')[['parent_QED','parent_SA_raw','parent_MW']].mean().mean()
        component=g.groupby('chemblid')[['active_QED','active_SA_raw','active_MW']].mean().mean()
        chemical.append(dict(study=study,group=group,n_outputs=len(g),n_targets=g.chemblid.nunique(),
             parent_QED=parent.parent_QED,active_QED=component.active_QED,
             parent_SA_raw=parent.parent_SA_raw,active_SA_raw=component.active_SA_raw,
             parent_MW=parent.parent_MW,active_MW=component.active_MW,
             salt_record_n=int((g.parent_fragment_removed==True).sum()),primary_B_selection_changed=False))
    pd.DataFrame(chemical).to_csv(ROOT/'chemical_form_group_sensitivity.csv',index=False)
    crows=[]
    for path in sorted((ROOT/'feedback/results').glob('*.json')):
        r=json.loads(path.read_text())
        crows.append(dict(slot_id=r['slot_id'],chemblid=r['chemblid'],branch=r['branch'],delivered=r['ok'],
             exact_copy_of_any_reference_component=(r['smiles'] in all_reference_fragments[(r['chemblid'],'M')]) if r['ok'] else None,
             original_prelocked_selection_unchanged=True))
    pd.DataFrame(crows).to_csv(ROOT/'C_reference_component_noncopy_sensitivity.csv',index=False)
    print(summary.to_string(index=False),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['calibration','scores','rmsd-test','reuse-b','form-sensitivity'])
    a=ap.parse_args()
    if a.phase=='rmsd-test':
        rmsd_fixture()
    elif a.phase=='reuse-b':
        reuse_b()
    elif a.phase=='form-sensitivity':
        form_sensitivity()
    else:
        globals()[a.phase]()
