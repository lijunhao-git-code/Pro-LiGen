"""Recompute current three-condition output novelty from released scientific records.

Usage: python audit_novelty.py --source /path/to/Pro-LiGen --out NEW_DIRECTORY
Requires NumPy and RDKit; no model API, training, docking, or network access.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,math
import numpy as np
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

SETS=['library','rc_triplet_inputs','logp_train','logp_validation','tpsa_train','tpsa_validation',
      'logp_inputs','tpsa_inputs','all_archived_gin_inputs']
CONDITION_SET={'ring':'rc_triplet_inputs','logp':'logp_inputs','tpsa':'tpsa_inputs'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvout(p,items):
    with p.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True);parser.add_argument('--out',required=True)
    parser.add_argument('--expected',help='Optional frozen audit directory; defaults to the released expected results if present')
    a=parser.parse_args();root=Path(a.source).resolve();out=Path(a.out).resolve()
    assert not out.exists(),'Use a new output directory to preserve prior audit records'
    out.mkdir(parents=True);RDLogger.DisableLog('rdApp.*')
    paths={
        'library':'data/retrieval/library.csv',
        'rc_triplet_inputs':'data/training/triplets_ring_count.csv',
        'logp_train':'extensions/20261102/gin/training/logp_train_triplets.csv',
        'logp_validation':'extensions/20261102/gin/training/logp_val_triplets.csv',
        'tpsa_train':'extensions/20261102/gin/training/tpsa_train_triplets.csv',
        'tpsa_validation':'extensions/20261102/gin/training/tpsa_val_triplets.csv',
        'ring_records':'extensions/20261102/lossless/ring_completed.json',
        'property_records':'extensions/20261102/lossless/properties_completed.json',
        'observations':'extensions/20261102/data/updated20261102/observations1500.csv',
        'ring_seeds':'data/SI_records/tasks25.json',
        'property_seeds':'extensions/20261102/gin/inputs/tasks25.json',
    }
    hashes={p:sha(root/p) for p in paths.values()}
    cache={}
    def molecule(s):
        if s in cache:return cache[s]
        m=Chem.MolFromSmiles(s);assert m is not None and m.GetNumAtoms()>0,s
        canonical=Chem.MolToSmiles(m,isomericSmiles=True)
        if canonical in cache:cache[s]=cache[canonical];return cache[s]
        free=Chem.Mol(m);Chem.RemoveStereochemistry(free)
        stereo_free=Chem.MolToSmiles(free,isomericSmiles=True)
        parts=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
        carbon=[p for p in parts if any(x.GetAtomicNum()==6 for x in p.GetAtoms())]
        component=m
        if len(parts)>1 and len(carbon)==1 and all(p.GetNumAtoms()==1 for p in parts if not any(x.GetAtomicNum()==6 for x in p.GetAtoms())):
            component=carbon[0]
        component_canonical=Chem.MolToSmiles(component,isomericSmiles=True)
        free_component=Chem.Mol(component);Chem.RemoveStereochemistry(free_component)
        scaf=MurckoScaffold.GetScaffoldForMol(m)
        Chem.RemoveStereochemistry(scaf)
        scaffold=Chem.MolToSmiles(scaf,isomericSmiles=True) if scaf.GetNumAtoms() else ''
        generic=MurckoScaffold.MakeScaffoldGeneric(scaf) if scaf.GetNumAtoms() else None
        generic_scaffold=Chem.MolToSmiles(generic,isomericSmiles=False) if generic is not None else ''
        val=dict(canonical=canonical,stereo_free=stereo_free,component=component_canonical,
            component_stereo_free=Chem.MolToSmiles(free_component,isomericSmiles=True),scaffold=scaffold,generic_scaffold=generic_scaffold,mol=m)
        cache[s]=cache[canonical]=val;return val

    librows=rows(root/paths['library']);assert len(librows)==26108
    library_ids={molecule(r['canonical'])['canonical']:r['query_id'] for r in librows}
    assert len(library_ids)==26108
    refs={'library':set(library_ids)};reference_counts={}
    for name in SETS[1:6]:
        source=rows(root/paths[name]);values=set()
        for row in source:
            for role in ['anchor','positive','negative']:values.add(molecule(row[role+'_smiles'])['canonical'])
        assert values<=refs['library'],name
        refs[name]=values;reference_counts[name]=len(source)
    refs['logp_inputs']=refs['logp_train']|refs['logp_validation']
    refs['tpsa_inputs']=refs['tpsa_train']|refs['tpsa_validation']
    refs['all_archived_gin_inputs']=set.union(*(refs[n] for n in SETS[1:6]))
    assert len(refs['rc_triplet_inputs'])==17325
    assert not (refs['logp_train']&refs['logp_validation'])
    assert not (refs['tpsa_train']&refs['tpsa_validation'])
    transformed={}
    for name,values in refs.items():
        transformed[name]={k:{molecule(s)[k] for s in values} for k in ['stereo_free','component','component_stereo_free','scaffold','generic_scaffold']}
        transformed[name]['scaffold'].discard('')
        transformed[name]['generic_scaffold'].discard('')
    seed_by_condition={}
    for condition,key in [('ring','ring_seeds'),('logp','property_seeds'),('tpsa','property_seeds')]:
        seed_by_condition[condition]={r['chemblid']:molecule(r['seed_smiles']) for r in json.loads((root/paths[key]).read_text(encoding='utf-8'))}
    seed_molecules={v['canonical'] for group in seed_by_condition.values() for v in group.values()}
    seed_scaffolds={molecule(s)['scaffold'] for s in seed_molecules}-{''}
    reference_rows=[]
    for name in SETS:
        reference_rows.append(dict(reference_set=name,unique_canonical_molecules=len(refs[name]),
            source_triplets=reference_counts.get(name,''),unique_nonempty_scaffolds=len(transformed[name]['scaffold']),
            unique_stereo_free=len(transformed[name]['stereo_free']),
            unique_atomic_component=len(transformed[name]['component']),
            source=paths.get(name,'union of named archived collections')))
    csvout(out/'reference_sets.csv',reference_rows)
    print('Reference sets prepared',len(refs['library']),len(refs['all_archived_gin_inputs']),flush=True)

    observed={r['slot_id']:r for r in rows(root/paths['observations'])}
    records=[]
    for label,key in [('ring','ring_records'),('properties','property_records')]:
        for raw in json.loads((root/paths[key]).read_text(encoding='utf-8')):
            cond='ring' if label=='ring' else raw['property'];sel=raw['selected_candidate'];sid=raw['slot_id']
            row=dict(condition=cond,group=raw['group'],chemblid=raw['chemblid'],slot_id=sid,position=raw['position'],
                delivered=int(sel is not None),canonical=sel['canonical'] if sel else '',
                source_json=paths[key],source_original_sha256=raw['source_sha256'])
            assert int(float(observed[sid]['S']))==row['delivered']
            if sel:
                assert molecule(sel['canonical'])['canonical']==sel['canonical']
                assert abs(float(observed[sid]['U'])-sel['QED'])<1e-12
            records.append(row)
    assert len(records)==1500 and len({r['slot_id'] for r in records})==1500
    assert sum(r['delivered'] for r in records)==1469
    records.sort(key=lambda r:(['ring','logp','tpsa'].index(r['condition']),r['group'],r['chemblid'],r['position']))
    distinct=sorted({r['canonical'] for r in records if r['delivered']})
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    libkeys=sorted(refs['library']);fps=[fpgen.GetFingerprint(molecule(s)['mol']) for s in libkeys]
    ref_indices={name:np.array([i for i,s in enumerate(libkeys) if s in refs[name]],dtype=int) for name in SETS}
    features={};nn_cache={}
    for i,canonical in enumerate(distinct):
        m=molecule(canonical);fp=fpgen.GetFingerprint(m['mol']);signature=fp.ToBinary()
        if signature not in nn_cache:
            sims=np.asarray(DataStructs.BulkTanimotoSimilarity(fp,fps),dtype=np.float64);nearest={}
            for name,indices in ref_indices.items():
                vals=sims[indices];best=float(vals.max());tied=indices[np.abs(vals-best)<=1e-12]
                idx=int(tied[0]);nearest[name]=(best,libkeys[idx],int(len(tied)))
            nn_cache[signature]=nearest
        f=dict(canonical=canonical,stereo_free=m['stereo_free'],atomic_component=m['component'],
            atomic_component_stereo_free=m['component_stereo_free'],scaffold=m['scaffold'],generic_scaffold=m['generic_scaffold'],has_scaffold=int(bool(m['scaffold'])),
            exact_any_seed=int(canonical in seed_molecules),scaffold_match_any_seed=int(m['scaffold'] in seed_scaffolds) if m['scaffold'] else None)
        for name in SETS:
            nn=nn_cache[signature][name]
            f.update({f'exact_{name}':int(canonical in refs[name]),
                f'stereo_free_match_{name}':int(m['stereo_free'] in transformed[name]['stereo_free']),
                f'atomic_component_match_{name}':int(m['component'] in transformed[name]['component']),
                f'atomic_component_stereo_free_match_{name}':int(m['component_stereo_free'] in transformed[name]['component_stereo_free']),
                f'scaffold_match_{name}':int(m['scaffold'] in transformed[name]['scaffold']) if m['scaffold'] else None,
                f'generic_scaffold_match_{name}':int(m['generic_scaffold'] in transformed[name]['generic_scaffold']) if m['generic_scaffold'] else None,
                f'nearest_tanimoto_{name}':nn[0],f'nearest_canonical_{name}':nn[1],f'nearest_ties_{name}':nn[2]})
        f['nearest_library_id']=library_ids[f['nearest_canonical_library']]
        features[canonical]=f
        if (i+1)%200==0:print('Nearest-neighbor queries',i+1,'/',len(distinct),flush=True)
    csvout(out/'unique_molecule_audit.csv',[features[s] for s in distinct])

    empty={k:None for k in next(iter(features.values()))}
    for row in records:
        if row['delivered']:row.update(features[row['canonical']])
        else:row.update(empty);row['canonical']=''
        name=CONDITION_SET[row['condition']]
        row['condition_gin_reference_set']=name
        seed=seed_by_condition[row['condition']][row['chemblid']]
        row['same_task_seed_scaffold']=int(row['scaffold']==seed['scaffold']) if row['delivered'] and row['has_scaffold'] else None
        row['same_task_seed_molecule']=int(row['canonical']==seed['canonical']) if row['delivered'] else None
        assert not row['same_task_seed_molecule']
        for metric in ['exact','scaffold_match','nearest_tanimoto']:
            row[metric+'_condition_gin_inputs']=row[metric+'_'+name]
    csvout(out/'final_records1500.csv',records)

    groups=defaultdict(list)
    for r in records:groups[(r['condition'],r['group'])].append(r)
    summary=[];targets=[];matrix=[]
    for (condition,group),planned in groups.items():
        selected=[r for r in planned if r['delivered']];unique={r['canonical'] for r in selected}
        scaffolds={r['scaffold'] for r in selected if r['has_scaffold']};ring=[r for r in selected if r['has_scaffold']]
        generic_scaffolds={r['generic_scaffold'] for r in ring}
        nn=np.array([r['nearest_tanimoto_library'] for r in selected])
        bytask=defaultdict(list)
        for r in planned:bytask[r['chemblid']].append(r)
        assert len(planned)==125 and len(bytask)==25
        target_values=[]
        for task,pr in sorted(bytask.items()):
            sr=[r for r in pr if r['delivered']];uq={r['canonical'] for r in sr}
            v=dict(condition=condition,group=group,chemblid=task,planned=len(pr),delivered=len(sr),unique_molecules=len(uq),
                uniqueness=len(uq)/len(sr) if sr else None,
                exact_library_hits=sum(r['exact_library'] for r in sr),
                exact_condition_gin_input_hits=sum(r['exact_condition_gin_inputs'] for r in sr),
                nearest_library_tanimoto_mean=float(np.mean([r['nearest_tanimoto_library'] for r in sr])) if sr else None)
            targets.append(v)
            if sr:target_values.append(v['nearest_library_tanimoto_mean'])
        row=dict(condition=condition,group=group,planned=len(planned),delivered=len(selected),
            unique_molecules=len(unique),uniqueness_pooled=len(unique)/len(selected),
            within_task_uniqueness_macro=float(np.mean([t['uniqueness'] for t in targets if t['condition']==condition and t['group']==group and t['uniqueness'] is not None])),
            exact_library_records=sum(r['exact_library'] for r in selected),
            exact_library_unique=sum(features[s]['exact_library'] for s in unique),
            novel_library_fraction_unique=1-sum(features[s]['exact_library'] for s in unique)/len(unique),
            exact_condition_gin_input_records=sum(r['exact_condition_gin_inputs'] for r in selected),
            exact_all_archived_gin_input_records=sum(r['exact_all_archived_gin_inputs'] for r in selected),
            nearest_library_tanimoto_mean_pooled=float(nn.mean()),nearest_library_tanimoto_mean_task_macro=float(np.mean(target_values)),
            nearest_library_tanimoto_median=float(np.median(nn)),nearest_library_tanimoto_q25=float(np.quantile(nn,.25)),
            nearest_library_tanimoto_q75=float(np.quantile(nn,.75)),nearest_library_tanimoto_min=float(nn.min()),nearest_library_tanimoto_max=float(nn.max()),
            ring_bearing_records=len(ring),acyclic_records=len(selected)-len(ring),unique_nonempty_scaffolds=len(scaffolds),
            library_unseen_unique_scaffolds=len(scaffolds-transformed['library']['scaffold']),
            library_unseen_scaffold_fraction=len(scaffolds-transformed['library']['scaffold'])/len(scaffolds) if scaffolds else None,
            condition_gin_unseen_unique_scaffolds=len(scaffolds-transformed[CONDITION_SET[condition]]['scaffold']),
            library_unseen_scaffold_records=sum(not r['scaffold_match_library'] for r in ring),
            library_and_seed_unseen_unique_scaffolds=len(scaffolds-transformed['library']['scaffold']-seed_scaffolds),
            unique_generic_scaffolds=len(generic_scaffolds),
            library_unseen_unique_generic_scaffolds=len(generic_scaffolds-transformed['library']['generic_scaffold']),
            same_task_seed_scaffold_records=sum(r['same_task_seed_scaffold'] for r in ring),
            stereo_free_library_hit_records=sum(r['stereo_free_match_library'] for r in selected),
            atomic_component_library_hit_records=sum(r['atomic_component_match_library'] for r in selected),
            component_stereo_free_library_hit_records=sum(r['atomic_component_stereo_free_match_library'] for r in selected))
        summary.append(row)
        for name in SETS:
            matrix.append(dict(condition=condition,group=group,reference_set=name,reference_unique_molecules=len(refs[name]),
                delivered_records=len(selected),unique_outputs=len(unique),
                exact_hit_records=sum(r['exact_'+name] for r in selected),exact_hit_unique=sum(features[s]['exact_'+name] for s in unique),
                stereo_free_hit_records=sum(r['stereo_free_match_'+name] for r in selected),
                atomic_component_hit_records=sum(r['atomic_component_match_'+name] for r in selected),
                atomic_component_stereo_free_hit_records=sum(r['atomic_component_stereo_free_match_'+name] for r in selected),
                nearest_tanimoto_mean=float(np.mean([r['nearest_tanimoto_'+name] for r in selected])),
                unique_nonempty_scaffolds=len(scaffolds),novel_unique_scaffolds=len(scaffolds-transformed[name]['scaffold'])))
    csvout(out/'group_summary12.csv',summary);csvout(out/'target_summary300.csv',targets);csvout(out/'reference_overlap_matrix108.csv',matrix)
    hits=[r for r in records if r['delivered'] and r['exact_library']]
    assert len(hits)==1 and hits[0]['slot_id']=='CHEMBL1821_5_G'
    csvout(out/'exact_library_hit_cases.csv',hits)
    sensitivity_hits=[r for r in records if r['delivered'] and r['atomic_component_stereo_free_match_library']]
    if sensitivity_hits:csvout(out/'representation_sensitivity_hit_cases.csv',sensitivity_hits)

    # Independent scalar pair checks of each reported best match, plus library self-match checks.
    for canonical in distinct[::max(1,len(distinct)//25)]:
        f=features[canonical]
        got=DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(molecule(canonical)['mol']),fpgen.GetFingerprint(molecule(f['nearest_canonical_library'])['mol']))
        assert abs(got-f['nearest_tanimoto_library'])<1e-14
    for idx in range(0,len(fps),max(1,len(fps)//11)):
        assert max(DataStructs.BulkTanimotoSimilarity(fps[idx],fps))==1.
    # Verify the full-library optimum independently for three fixed output structures.
    independent=[]
    for canonical in distinct[::max(1,len(distinct)//3)][:3]:
        query=fpgen.GetFingerprint(molecule(canonical)['mol'])
        values=[DataStructs.TanimotoSimilarity(query,fp) for fp in fps]
        best=max(values);tie=[i for i,x in enumerate(values) if abs(x-best)<=1e-12]
        assert abs(best-features[canonical]['nearest_tanimoto_library'])<1e-14
        assert libkeys[tie[0]]==features[canonical]['nearest_canonical_library']
        independent.append(dict(canonical=canonical,all_library_pairs_checked=len(fps),maximum=best,ties=len(tie)))
    assert all(sha(root/name)==h for name,h in hashes.items())
    sourceproof=dict(input_sha256=hashes,RDKit=rdBase.rdkitVersion,numpy=np.__version__,
        fingerprint=dict(type='binary Morgan',radius=2,bits=2048,includeChirality=False,metric='Tanimoto'),
        canonicalization='RDKit isomeric canonical SMILES; retain charge, isotopes, components and tautomers',
        stereo_free='RemoveStereochemistry then canonical isomeric SMILES; retain isotopes and charge',
        atomic_component_sensitivity='Retain the sole carbon-containing component only when all other components are single atoms; otherwise leave unchanged',
        scaffold='Typed Bemis-Murcko scaffold without stereochemistry; empty acyclic scaffolds excluded from scaffold novelty denominator',
        generic_framework_sensitivity='RDKit MakeScaffoldGeneric applied to the Murcko scaffold; atom identities and bond orders generalized',
        seed_scaffold_sensitivity='Check typed scaffolds against reference-library and all25 task-seed scaffolds jointly; exact own-seed scaffold retention also counted',
        nearest_tie_break='ascending isomeric canonical reference string; ties within1e-12',
        rc_input_scope='all participants in archived RC triplets',
        property_input_scope='training and validation participants reported separately and as property-specific unions',
        novelty_scope='released reference collections, not global chemical novelty',statistical_tests_added=0)
    dump(out/'protocol_and_sources.json',sourceproof)
    overall=dict(planned=1500,delivered=1469,unique_molecules=len(distinct),exact_library_hit_records=len(hits),
        exact_library_hit_unique=sum(features[s]['exact_library'] for s in distinct),
        exact_all_archived_gin_input_records=sum(r['exact_all_archived_gin_inputs'] for r in records if r['delivered']),
        stereo_free_library_hit_records=sum(r['stereo_free_match_library'] for r in records if r['delivered']),
        atomic_component_library_hit_records=sum(r['atomic_component_match_library'] for r in records if r['delivered']),
        component_stereo_free_library_hit_records=sum(r['atomic_component_stereo_free_match_library'] for r in records if r['delivered']),
        acyclic_records=sum(not r['has_scaffold'] for r in records if r['delivered']),
        unique_nonempty_scaffolds=len({features[s]['scaffold'] for s in distinct if features[s]['has_scaffold']}),
        library_unseen_unique_scaffolds=len({features[s]['scaffold'] for s in distinct if features[s]['has_scaffold']}-transformed['library']['scaffold']),
        library_unseen_scaffold_records=sum(not r['scaffold_match_library'] for r in records if r['delivered'] and r['has_scaffold']))
    all_scaffolds={features[s]['scaffold'] for s in distinct if features[s]['has_scaffold']}
    all_generic={features[s]['generic_scaffold'] for s in distinct if features[s]['has_scaffold']}
    overall.update(exact_any_seed_records=sum(r['exact_any_seed'] for r in records if r['delivered']),
        library_and_seed_unseen_unique_scaffolds=len(all_scaffolds-transformed['library']['scaffold']-seed_scaffolds),
        unique_generic_scaffolds=len(all_generic),library_unseen_unique_generic_scaffolds=len(all_generic-transformed['library']['generic_scaffold']),
        same_task_seed_scaffold_records=sum(r['same_task_seed_scaffold'] for r in records if r['delivered'] and r['has_scaffold']))
    dump(out/'overall_summary.json',overall)
    expected=Path(a.expected).resolve() if a.expected else root/'extensions/novelty20261102/results'
    csv_checks=[]
    if expected.exists():
        for candidate in sorted(out.glob('*.csv')):
            reference=expected/candidate.name;assert reference.exists(),reference
            actual_rows=rows(candidate);expected_rows=rows(reference)
            assert len(actual_rows)==len(expected_rows),(candidate.name,'row count')
            for number,(actual,old) in enumerate(zip(actual_rows,expected_rows),1):
                assert actual.keys()==old.keys(),(candidate.name,'columns')
                for key,value in actual.items():
                    if value==old[key]:continue
                    numeric=any(x in key for x in ['tanimoto','_fraction','uniqueness']) and not any(x in key for x in ['canonical','source'])
                    assert numeric and math.isclose(float(value),float(old[key]),rel_tol=1e-11,abs_tol=1e-12),(candidate.name,number,key,value,old[key])
            csv_checks.append(dict(file=candidate.name,rows=len(actual_rows),match=True))
    dump(out/'verification.json',dict(status='passed',source_files_unchanged=True,planned_records=1500,delivered_records=1469,
        unique_outputs=len(distinct),group_rows=len(summary),task_group_rows=len(targets),reference_matrix_rows=len(matrix),
        all_archived_GIN_molecules_are_in_reference_library=True,property_train_validation_exact_disjoint=True,
        exact_hits_match_previously_recorded_case=True,scalar_NN_maximum_checks=independent,
        frozen_csv_comparisons=csv_checks,
        fingerprint_cached_queries=len(nn_cache),model_API_calls=0,new_training_jobs=0,new_docking_jobs=0))
    print(json.dumps(dict(status='passed',**overall),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
