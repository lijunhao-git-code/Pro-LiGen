"""Recompute GIN embeddings, Morgan scores, common pools and all2,250 references."""
import os,time,hashlib
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd,torch
from torch_geometric.data import Batch
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from common import ROOT,PROPERTIES,read,read_jsonl,sha,now
from gin_model_original import GINEncoder,mol_to_graph
from analysis_utils import save_json


def main():
    started=time.monotonic();RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    lib=pd.read_csv(ROOT/'inputs/library.csv');keys=lib.canonical.to_numpy(str);tasks=read(ROOT/'inputs/tasks25.json')
    entries=read_jsonl(ROOT/'inputs/entries1000.jsonl');refs=pd.read_csv(ROOT/'inputs/references2250.csv')
    pools={(r['property'],r['chemblid']):r for r in pd.read_csv(ROOT/'inputs/reference_pools50.csv').to_dict('records')}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    fps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in keys];counts=np.array([f.GetNumOnBits() for f in fps])
    graphs=[mol_to_graph(s) for s in keys];device='cuda' if torch.cuda.is_available() else 'cpu'
    zall={};z_cached={};models={};embdiff={}
    for prop in PROPERTIES:
        model=GINEncoder(84,256,256,5).to(device)
        model.load_state_dict(torch.load(ROOT/f'models/{prop}_best.pth',map_location=device,weights_only=True),strict=True);model.eval();models[prop]=model
        pieces=[]
        with torch.inference_mode():
            for offset in range(0,len(graphs),128):
                raw=model(Batch.from_data_list(graphs[offset:offset+128]).to(device))
                pieces.append((raw/raw.norm(dim=1,keepdim=True)).cpu().numpy())
        z=np.concatenate(pieces);old=np.load(ROOT/f'inputs/{prop}_library_embeddings.npy')
        assert np.allclose(z,old,atol=2e-6,rtol=0)
        embdiff[prop]=float(np.max(abs(z-old)));zall[prop]=z;z_cached[prop]=old
        print('VERIFIED EMBEDDINGS',prop,len(z),flush=True)
    ranked={};cached_ranked={};scores_by_task={};checks=0;differences=[];fixed_g=set();cached_differences=[]
    for task in tasks:
        seed=task['seed_smiles'];fp=fpgen.GetFingerprint(Chem.MolFromSmiles(seed));n=fp.GetNumOnBits()
        tan=np.array(DataStructs.BulkTanimotoSimilarity(fp,fps));shared=np.rint(tan*(n+counts)/(1+tan))
        m=shared/np.sqrt(n*counts)
        for prop,spec in PROPERTIES.items():
            center=task[prop]+spec['delta'];mask=(abs(lib[prop].to_numpy()-center)<=spec['tolerance']+1e-8)&(keys!=seed)
            indices=np.flatnonzero(mask);pool=pools[(prop,task['chemblid'])]
            assert len(indices)==pool['pool_size']
            assert hashlib.sha256('\n'.join(sorted(keys[indices])).encode()).hexdigest()==pool['pool_hash']
            with torch.inference_mode():
                q=models[prop](Batch.from_data_list([mol_to_graph(seed)]).to(device));q=(q/q.norm(dim=1,keepdim=True)).cpu().numpy()[0]
            scores=zall[prop]@q
            cached_scores=z_cached[prop]@q
            cached_ranked[(prop,task['chemblid'])]=list(keys[indices[np.lexsort((keys[indices],-cached_scores[indices]))[:3]]])
            scores_by_task[(prop,task['chemblid'])]=(scores,indices)
            # Full sorting, independent of original partition/topk implementation.
            ranked[(prop,task['chemblid'],'M')]=list(keys[indices[np.lexsort((keys[indices],-m[indices]))[:3]]])
            ranked[(prop,task['chemblid'],'G')]=list(keys[indices[np.lexsort((keys[indices],-scores[indices]))[:3]]])
            for repeat in range(1,6):
                seednum=int.from_bytes(hashlib.sha256(f'42|{prop}|{task["chemblid"]}|{repeat}'.encode()).digest()[:16],'big')
                ranked[(prop,task['chemblid'],'R',repeat)]=list(keys[np.random.default_rng(seednum).choice(indices,3,replace=False)])
    for entry in entries:
        if entry['group']=='T':assert entry['exemplars']==[];continue
        key=(entry['property'],entry['chemblid'],entry['group'])
        expected=ranked[key+(entry['position'],)] if entry['group']=='R' else ranked[key]
        if entry['group']=='G' and entry['position']==1 and cached_ranked[(entry['property'],entry['chemblid'])]!=entry['exemplars']:
            cached_differences.append(dict(property=entry['property'],chemblid=entry['chemblid'],frozen=entry['exemplars'],
                reselected=cached_ranked[(entry['property'],entry['chemblid'])]))
        if expected!=entry['exemplars']:
            assert entry['group']=='G',entry['slot_id']
            scores,indices=scores_by_task[(entry['property'],entry['chemblid'])]
            fixed_indices=np.array([int(np.flatnonzero(keys==s)[0]) for s in entry['exemplars']])
            cutoff=float(np.sort(scores[indices])[-3])
            deficit=max(0.,float(cutoff-min(scores[fixed_indices])))
            # Validation tolerance for float32 graph reduction/matmul, never a new selection rule.
            assert deficit<1e-5,(entry['slot_id'],deficit)
            taskkey=(entry['property'],entry['chemblid'])
            if taskkey not in fixed_g:
                differences.append(dict(property=entry['property'],chemblid=entry['chemblid'],
                    frozen_examples=entry['exemplars'],fresh_examples=expected,score_deficit=deficit,
                    explanation='Near-tied float32 GIN scores differ under fresh graph reductions. Actual pre-generation frozen examples remain authoritative and unchanged.'))
                fixed_g.add(taskkey)
        part=refs[refs.slot_id.eq(entry['slot_id'])].sort_values('rank')
        assert part.smiles.tolist()==entry['exemplars'];checks+=len(expected)
    assert checks==2250
    result=dict(status='passed_exact' if not differences else 'passed_with_documented_float32_near_ties',time_utc=now(),
        all_reference_identities_reselected_exact=not differences,reference_records=checks,
        Morgan_and_random_reselected_exact=True,frozen_reference_table_matches_all_actual_prompts_exact=True,
        all_GIN_reselected_exact_from_saved_library_embeddings=not cached_differences,
        cached_embedding_reselection_differences=cached_differences,
        saved_embedding_sha256={p:sha(ROOT/f'inputs/{p}_library_embeddings.npy') for p in PROPERTIES},
        near_tie_task_count=len(differences),near_tie_details=differences,numeric_validation_tolerance=1e-5,
        common_pools_verified=50,library_embeddings_recomputed=2*len(lib),embedding_max_abs_diff=embdiff,
        inference_uses_saved_best_weights=True,full_library_sort_not_partition_topk=True,
        fresh_R_sampling_matches_fixed_seeds=True,no_new_API_calls=True,elapsed_seconds=round(time.monotonic()-started,3),
        code_sha256=sha(__file__))
    save_json(ROOT/'analysis/retrieval_verification.json',result);print('RETRIEVAL VERIFICATION',result,flush=True)


if __name__=='__main__':main()
