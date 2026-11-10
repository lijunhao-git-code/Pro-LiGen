"""Audit molecular input curation and dataset relationships without model calls.

python audit_data_curation.py --source /path/to/Pro-LiGen --task-sources TASK_SOURCES --out NEW_DIRECTORY
Requires NumPy and RDKit 2026.3.4. Sources are read only.
"""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,hashlib,json,random
import numpy as np
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def dump(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(p,data):
    assert data,p
    with p.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True,type=Path)
    ap.add_argument('--task-sources',type=Path,default=Path(__file__).resolve().parents[1]/'data/curation20261102/task_sources')
    ap.add_argument('--out',required=True,type=Path)
    ap.add_argument('--expected',type=Path,help='Optional frozen results for per-field CSV verification')
    a=ap.parse_args();root=a.source.resolve();ts=a.task_sources.resolve();out=a.out.resolve()
    assert not out.exists(),'Use a new output directory';out.mkdir(parents=True)
    RDLogger.DisableLog('rdApp.*');cache={}
    def mol(s):
        if s in cache:return cache[s]
        m=Chem.MolFromSmiles(s)
        if m is None or not m.GetNumAtoms():return None
        c=Chem.MolToSmiles(m,isomericSmiles=True)
        if c in cache:cache[s]=cache[c];return cache[c]
        no=Chem.Mol(m);Chem.RemoveStereochemistry(no)
        parts=Chem.GetMolFrags(m,asMols=True,sanitizeFrags=True)
        carb=[p for p in parts if any(x.GetAtomicNum()==6 for x in p.GetAtoms())]
        comp=m
        if len(parts)>1 and len(carb)==1 and all(p.GetNumAtoms()==1 for p in parts if not any(x.GetAtomicNum()==6 for x in p.GetAtoms())):comp=carb[0]
        ns=Chem.Mol(comp);Chem.RemoveStereochemistry(ns)
        scaffold=MurckoScaffold.GetScaffoldForMol(m);Chem.RemoveStereochemistry(scaffold)
        v=dict(canonical=c,stereo_free=Chem.MolToSmiles(no,isomericSmiles=True),
            component=Chem.MolToSmiles(comp,isomericSmiles=True),component_stereo_free=Chem.MolToSmiles(ns,isomericSmiles=True),
            scaffold=Chem.MolToSmiles(scaffold,isomericSmiles=True) if scaffold.GetNumAtoms() else '',
            RC=m.GetRingInfo().NumRings(),MW=Descriptors.MolWt(m),fragments=len(parts),carbon=bool(carb),mol=m)
        cache[c]=cache[s]=v;return v
    paths={
        'raw_library':'data/training/source_smiles26402.csv','library':'data/retrieval/library.csv',
        'source_mapping':'data/task_sources/best_ligand.csv','parent_seeds':'data/SI_records/tasks25.json',
        'used_seeds':'extensions/20261102/gin/inputs/tasks25.json',
        'property_library':'extensions/20261102/gin/inputs/library.csv',
        'rc_inputs':'data/training/triplets_ring_count.csv',
        'logp_train':'extensions/20261102/gin/training/logp_train_triplets.csv',
        'logp_validation':'extensions/20261102/gin/training/logp_val_triplets.csv',
        'tpsa_train':'extensions/20261102/gin/training/tpsa_train_triplets.csv',
        'tpsa_validation':'extensions/20261102/gin/training/tpsa_val_triplets.csv',
        'training_implementation':'archive/train_gin.py',
        'task_assembly':'archive/study_a/build_prompts_rc_expand.py',
        'initial_task_adoption':'archive/generation/prepare_inputs.py',
    }
    hashes={v:sha(root/v) for v in paths.values()}
    task_hashes={p.name:sha(p) for p in sorted(ts.iterdir()) if p.is_file()}
    raw=rows(root/paths['raw_library']);rawcanon=[];invalid=[]
    for i,r in enumerate(raw):
        v=mol(r['smiles'])
        if v is None:invalid.append(i)
        else:rawcanon.append(v['canonical'])
    lib=rows(root/paths['library']);libmap={r['canonical']:r for r in lib}
    assert len(lib)==len(libmap)==26108
    assert set(rawcanon)==set(libmap)
    assert all(mol(s)['canonical']==s for s in libmap)
    single={s for s in libmap if mol(s)['fragments']==1 and mol(s)['carbon']}
    prop=rows(root/paths['property_library'])
    assert len(prop)==len({r['canonical'] for r in prop})==24693
    assert single=={r['canonical'] for r in prop}
    assert all(r['split']==libmap[r['canonical']]['split'] for r in prop)
    sets={'library':set(libmap),'single_component_library':single}
    for split in ['train','val','test']:
        sets['library_'+split]={s for s,r in libmap.items() if r['split']==split}
    assert set.union(*(sets['library_'+s] for s in ['train','val','test']))==sets['library']
    for x,y in [('train','val'),('train','test'),('val','test')]:assert not sets['library_'+x]&sets['library_'+y]
    triplets={};triplet_n={}
    for name in ['rc_inputs','logp_train','logp_validation','tpsa_train','tpsa_validation']:
        rr=rows(root/paths[name]);triplet_n[name]=len(rr);values=set();triplets[name]=rr
        for r in rr:
            cs=[mol(r[role+'_smiles'])['canonical'] for role in ['anchor','positive','negative']]
            values.update(cs)
            if name!='rc_inputs':
                split='train' if name.endswith('_train') else 'val'
                assert all(s in sets['library_'+split] for s in cs),(name,r)
        assert values<=sets['library'];sets[name]=values
    sets['all_gin_inputs']=set.union(*(sets[n] for n in triplets))
    parent=read(root/paths['parent_seeds']);used=read(root/paths['used_seeds'])
    pd={r['chemblid']:r for r in parent};ud={r['chemblid']:r for r in used}
    assert len(pd)==len(ud)==25 and set(pd)==set(ud)
    sets['parent_seeds']={mol(r['seed_smiles'])['canonical'] for r in parent}
    sets['used_seeds']={mol(r['seed_smiles'])['canonical'] for r in used}
    mapping=rows(root/paths['source_mapping']);mapping_by_id=defaultdict(list)
    mapping_invalid=[]
    for i,r in enumerate(mapping):
        mapping_by_id[r['chemblid']].append((i,r))
        if mol(r['best_ligand_smiles']) is None:mapping_invalid.append(dict(row_zero_based=i,chemblid=r['chemblid']))
    lists=read(ts/'task_list_provenance.json')['task_lists']
    old25=lists['exp30v2_pg_mw.jsonl']['targets'];old19=lists['exp30rc_pg_aggr.jsonl']['targets'];new6=lists['rc_expand_new6.jsonl']['targets']
    assert set(old19)<=set(old25) and not set(new6)&set(old25)
    assert set(pd)==set(old19)|set(new6)==set(lists['exp30rc25_pg_mild.jsonl']['targets'])
    retrieval=rows(ts/'retrieved_rc_k60.csv');retrievalmap={r['chemblid']:r for r in retrieval}
    # The historical table contains repeated target mappings. Preserve its row
    # order and multiplicity when reproducing the original random.sample call.
    assert all((r['status'],r['best_ligand_smiles'])==(retrievalmap[r['chemblid']]['status'],retrievalmap[r['chemblid']]['best_ligand_smiles']) for r in retrieval)
    assert set(old19)=={c for c in old25 if retrievalmap.get(c,{}).get('status')=='ok'}
    candidates=[];stages=Counter()
    for r in retrieval:
        if r.get('status')!='ok':continue
        stages['retrieval_ok']+=1;c=r['chemblid']
        if c in set(old25) or c=='CHEMBL3401':continue
        stages['not_original_panel_or_named_exclusion']+=1
        v=mol(r['best_ligand_smiles'])
        if v is None or v['RC']<2:continue
        stages['seed_at_least_two_rings']+=1
        if v['MW']>700:continue
        stages['seed_MW_at_most_700']+=1;candidates.append(c)
    picked=random.Random(42).sample(candidates,6)
    assert sorted(picked)==sorted(new6),(picked,new6)
    tasks=[]
    for c in sorted(pd):
        p=mol(pd[c]['seed_smiles']);u=mol(ud[c]['seed_smiles'])
        matches=[i for i,r in mapping_by_id[c] if mol(r['best_ligand_smiles'])['canonical']==p['canonical'] and r['uniprotid']==pd[c]['uniprotid']]
        assert matches,(c,mapping_by_id[c]);assert p['RC']>=2
        tasks.append(dict(chemblid=c,uniprotid=pd[c]['uniprotid'],source_mapping_rows_zero_based=';'.join(map(str,matches)),
            task_origin='retained_ring_feasible' if c in set(old19) else 'seed42_addition',
            parent_canonical=p['canonical'],used_canonical=u['canonical'],form_changed=int(p['canonical']!=u['canonical']),
            seed_RC=p['RC'],parent_MW=p['MW'],used_MW=u['MW'],
            parent_exact_library=int(p['canonical'] in sets['library']),used_exact_library=int(u['canonical'] in sets['library']),
            parent_exact_gin=int(p['canonical'] in sets['all_gin_inputs']),used_exact_gin=int(u['canonical'] in sets['all_gin_inputs'])))
    save(out/'task_inventory25.csv',tasks)
    save(out/'task_selection_flow.csv',[dict(stage=k,rows=v) for k,v in dict(source_mapping=len(mapping),legacy_retrieval_rows=len(retrieval),original_panel=len(old25),retained_ring_feasible=len(old19),**stages,random_additions=len(new6),final_fixed_panel=len(pd)).items()])
    transformed={n:{k:{mol(s)[k] for s in values}-{''} for k in ['canonical','stereo_free','component','component_stereo_free','scaffold']} for n,values in sets.items()}
    setrows=[];membership=[]
    for n,values in sets.items():
        setrows.append(dict(dataset=n,unique_molecules=len(values),unique_stereo_free=len(transformed[n]['stereo_free']),
            unique_components=len(transformed[n]['component']),unique_scaffolds=len(transformed[n]['scaffold']),source_triplets=triplet_n.get(n,'')))
        membership.extend(dict(dataset=n,canonical=s,library_id=libmap[s]['query_id'] if s in libmap else '') for s in sorted(values))
    save(out/'set_inventory.csv',setrows);save(out/'set_membership.csv',membership)
    pairs=[];names=list(sets)
    for i,n in enumerate(names):
        for m in names[i+1:]:
            pairs.append(dict(set_a=n,set_b=m,n_a=len(sets[n]),n_b=len(sets[m]),
                exact_intersection=len(sets[n]&sets[m]),
                stereo_free_shared_keys=len(transformed[n]['stereo_free']&transformed[m]['stereo_free']),
                component_shared_keys=len(transformed[n]['component']&transformed[m]['component']),
                component_stereo_free_shared_keys=len(transformed[n]['component_stereo_free']&transformed[m]['component_stereo_free']),
                scaffold_shared_keys=len(transformed[n]['scaffold']&transformed[m]['scaffold'])))
    save(out/'pairwise_set_overlap.csv',pairs)
    for p in ['logp','tpsa']:assert not sets[p+'_train']&sets[p+'_validation']
    print('Curation and selection verified; starting cross-set nearest-neighbor audit.',flush=True)
    # Compute each query against the entire canonical library once, then obtain subset maxima.
    searches=[('parent_seeds','library'),('used_seeds','single_component_library'),('parent_seeds','rc_inputs'),
        ('used_seeds','logp_train'),('used_seeds','logp_validation'),('used_seeds','tpsa_train'),('used_seeds','tpsa_validation'),
        ('library_val','library_train'),('library_test','library_train'),('logp_validation','logp_train'),('tpsa_validation','tpsa_train')]
    queries=defaultdict(list)
    for q,r in searches:
        for s in sets[q]:queries[s].append((q,r))
    keys=sorted(sets['library']);gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    fps=[gen.GetFingerprint(mol(s)['mol']) for s in keys];mask={n:np.array([i for i,s in enumerate(keys) if s in values],dtype=int) for n,values in sets.items() if values<=sets['library']}
    nn=[];cachefp={}
    for i,s in enumerate(sorted(queries)):
        fp=gen.GetFingerprint(mol(s)['mol']);sig=fp.ToBinary();wanted={r for _,r in queries[s]}
        if sig not in cachefp:cachefp[sig]={}
        missing=wanted-set(cachefp[sig])
        if missing:
            sims=np.array(DataStructs.BulkTanimotoSimilarity(fp,fps),dtype=np.float64)
            for r in missing:
                ix=mask[r];vals=sims[ix];best=float(vals.max());tied=ix[np.abs(vals-best)<=1e-12];nearest=keys[int(tied[0])]
                cachefp[sig][r]=(best,nearest,len(tied))
        for q,r in queries[s]:
            best,nearest,ties=cachefp[sig][r]
            nn.append(dict(query_set=q,reference_set=r,canonical=s,nearest_canonical=nearest,nearest_library_id=libmap[nearest]['query_id'],
                nearest_tanimoto=best,nearest_ties=ties,exact_in_reference=int(s in sets[r]),
                scaffold_in_reference=int(mol(s)['scaffold'] in transformed[r]['scaffold']) if mol(s)['scaffold'] else ''))
        if (i+1)%1000==0:print('Nearest-neighbor queries',i+1,'/',len(queries),flush=True)
    nn.sort(key=lambda r:(r['query_set'],r['reference_set'],r['canonical']));save(out/'cross_set_nearest_neighbors.csv',nn)
    summaries=[]
    for q,r in searches:
        rr=[v for v in nn if v['query_set']==q and v['reference_set']==r];v=np.array([x['nearest_tanimoto'] for x in rr]);assert len(rr)==len(sets[q])
        summaries.append(dict(query_set=q,reference_set=r,n_query=len(rr),n_reference=len(sets[r]),exact_shared_molecules=len(sets[q]&sets[r]),
            nearest_median=float(np.median(v)),nearest_q25=float(np.quantile(v,.25)),nearest_q75=float(np.quantile(v,.75)),
            nearest_ge080=int((v>=.8-1e-12).sum()),nearest_ge090=int((v>=.9-1e-12).sum()),nearest_equal1=int((v>=1-1e-12).sum()),
            query_scaffold_shared_records=sum(x['scaffold_in_reference']==1 for x in rr)))
    save(out/'cross_set_nearest_summary.csv',summaries)
    save(out/'near_duplicate_cases.csv',[r for r in nn if r['nearest_tanimoto']>=.8-1e-12])
    # Independent scalar maximum verification on one query from each training/validation pair.
    scalar=[]
    for q,r in [('logp_validation','logp_train'),('tpsa_validation','tpsa_train')]:
        row=next(x for x in nn if x['query_set']==q and x['reference_set']==r)
        fp=gen.GetFingerprint(mol(row['canonical'])['mol']);maximum=max(DataStructs.TanimotoSimilarity(fp,fps[int(i)]) for i in mask[r])
        assert abs(maximum-row['nearest_tanimoto'])<1e-14;scalar.append(dict(query_set=q,pairs_checked=len(mask[r]),maximum=maximum))
    counts=Counter(rawcanon);source_stats=dict(source_rows=len(raw),parseable_rows=len(rawcanon),invalid_rows=len(invalid),
        unique_canonical=len(set(rawcanon)),duplicate_rows_removed=len(rawcanon)-len(set(rawcanon)),
        duplicated_canonical_keys=sum(n>1 for n in counts.values()),single_component_carbon_library=len(single),
        omitted_from_continuous_library=len(lib)-len(single),source_mapping_rows=len(mapping),final_task_count=25,
        legacy_original_panel=25,retained_ring_feasible=19,seed42_additions=6,addition_candidate_count=len(candidates),
        addition_unique_candidate_targets=len(set(candidates)),retrieval_rows=len(retrieval),retrieval_unique_target_ids=len(retrievalmap),
        source_mapping_unique_target_ids=len(mapping_by_id),source_mapping_unparseable_rows=mapping_invalid,
        selected_additions=sorted(picked),changed_task_seed_forms=[r['chemblid'] for r in tasks if r['form_changed']],
        parent_seed_exact_library=sum(r['parent_exact_library'] for r in tasks),used_seed_exact_library=sum(r['used_exact_library'] for r in tasks),
        parent_seed_exact_gin=sum(r['parent_exact_gin'] for r in tasks),used_seed_exact_gin=sum(r['used_exact_gin'] for r in tasks))
    dump(out/'curation_summary.json',source_stats)
    protocol=dict(rdkit=rdBase.rdkitVersion,numpy=np.__version__,source_files=hashes,task_source_files=task_hashes,
        canonical='Isomeric canonical SMILES, preserving charge, components, isotopes and specified stereo; no tautomer canonicalization or neutralization',
        component_rule='Keep sole carbon-containing component only when every accompanying component is a single atom; otherwise keep original form',
        scaffold='Typed Bemis-Murcko scaffold without stereochemistry; omit empty scaffolds',
        fingerprint=dict(kind='binary Morgan',radius=2,bits=2048,includeChirality=False,similarity='Tanimoto'),
        near_duplicate_diagnostics='Report NN Tanimoto >=0.80 and >=0.90 descriptively; these are not original exclusion rules and do not change memberships',
        library_split_scope='Canonical molecule assignments in library.csv; not the historical RC trainer triplet split',
        rc_scope='All archived triplet participants; original unseeded train/validation index assignment is not represented as recovered',
        initializer_scope='Audit enumerated archived molecular inputs; do not infer the input membership of an earlier initializer from its checkpoint',
        task_selection='Retain19 retrieval-feasible legacy tasks; sample6 from original ordered retrieval candidates after exclusions with Python random.Random(42); freeze final25 for subsequent studies',
        task_constraints='RC>=2; legacy retrieval status ok; added tasks additionally exclude old25 and CHEMBL3401 and require seed MW<=700. Do not retroactively apply added-task MW filter to retained19.',
        no_structure_or_data_changes=True,new_generation_calls=0,new_training_jobs=0,new_docking_jobs=0)
    dump(out/'protocol_and_sources.json',protocol)
    assert all(sha(root/p)==v for p,v in hashes.items())
    assert all(sha(ts/p)==v for p,v in task_hashes.items())
    comparisons=[]
    if a.expected:
        for f in sorted(a.expected.glob('*.csv')):
            got=rows(out/f.name);expected=rows(f);assert len(got)==len(expected),f.name
            for g,e in zip(got,expected):
                assert g.keys()==e.keys(),f.name
                for k,v in e.items():
                    if v==g[k]:continue
                    try:assert abs(float(v)-float(g[k]))<1e-12,(f.name,k,v,g[k])
                    except ValueError:raise AssertionError((f.name,k,v,g[k]))
            comparisons.append(dict(file=f.name,rows=len(got),match=True))
    verification=dict(status='passed',source_files_unchanged=True,source_library_sets_equal=True,
        single_component_subset_matches_existing_property_inputs=True,all_task_seed_mappings_verified=True,
        selected_additions_exactly_reproduced=True,task_panel_matches_frozen_inputs=True,
        library_canonical_splits_exact_disjoint=True,property_triplets_respect_existing_molecule_splits=True,
        property_train_validation_molecular_intersection_zero=True,sets=len(sets),pairwise_rows=len(pairs),
        nearest_summary_rows=len(summaries),nearest_query_rows=len(nn),nearest_scalar_checks=scalar,
        frozen_csv_comparisons=comparisons,new_API_calls=0,new_training_jobs=0,new_docking_jobs=0)
    dump(out/'verification.json',verification)
    print(json.dumps(dict(status='passed',curation=source_stats,sets=len(sets),pairs=len(pairs)),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
