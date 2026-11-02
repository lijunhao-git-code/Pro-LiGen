"""Reselect the extended-study M/G/R references from released inputs and frozen weights."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys
os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np
import torch
from torch_geometric.data import Batch
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1];X=ROOT/'extensions/20261102';G=X/'gin'
sys.path.insert(0,str(G/'code'))
from gin_model_original import GINEncoder,mol_to_graph
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--recompute-library',action='store_true');a=p.parse_args()
    out=Path(a.out).resolve();assert not out.exists(),'Use a new output directory';out.mkdir(parents=True)
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    lib=rows(G/'inputs/library.csv');keys=np.array([r['canonical'] for r in lib]);assert len(lib)==24693
    tasks=json.loads((G/'inputs/tasks25.json').read_text(encoding='utf-8'));refs=rows(G/'inputs/references2250.csv')
    pools={(r['property'],r['chemblid']):r for r in rows(G/'inputs/reference_pools50.csv')}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    fps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in keys];counts=np.array([v.GetNumOnBits() for v in fps])
    values={pr:np.array([float(r[pr]) for r in lib]) for pr in ['logp','tpsa']};expected={}
    for r in refs:expected.setdefault((r['property'],r['chemblid'],r['group'],int(r['position'])),[]).append((int(r['rank']),r['smiles']))
    expected={k:[s for _,s in sorted(v)] for k,v in expected.items()}
    near=[];selected=[];exact=0;checked=0;embedding_diff={};models={};embeddings={}
    for prop in ['logp','tpsa']:
        model=GINEncoder(84,256,256,5).cpu();model.load_state_dict(torch.load(G/f'models/{prop}_best.pth',map_location='cpu',weights_only=True),strict=True);model.eval();models[prop]=model
        saved=np.load(G/f'inputs/{prop}_library_embeddings.npy');assert saved.shape==(24693,256)
        if a.recompute_library:
            parts=[]
            with torch.inference_mode():
                for start in range(0,len(keys),128):
                    z=model(Batch.from_data_list([mol_to_graph(s) for s in keys[start:start+128]]));parts.append((z/z.norm(dim=1,keepdim=True)).numpy())
            fresh=np.concatenate(parts);embedding_diff[prop]=float(abs(fresh-saved).max());assert np.allclose(fresh,saved,atol=2e-6,rtol=0)
            embeddings[prop]=fresh
        else:embeddings[prop]=saved
        print('Ready:',prop,flush=True)
    for t in tasks:
        seed=t['seed_smiles'];fp=fpgen.GetFingerprint(Chem.MolFromSmiles(seed));n=fp.GetNumOnBits()
        tan=np.array(DataStructs.BulkTanimotoSimilarity(fp,fps));shared=np.rint(tan*(n+counts)/(1+tan));m=shared/np.sqrt(n*counts)
        for prop,delta,tolerance in [('logp',-1.,.2),('tpsa',-20.,5.)]:
            indices=np.flatnonzero((abs(values[prop]-(float(t[prop])+delta))<=tolerance+1e-8)&(keys!=seed))
            pool=pools[(prop,t['chemblid'])];assert len(indices)==int(pool['pool_size'])
            assert hashlib.sha256('\n'.join(sorted(keys[indices])).encode()).hexdigest()==pool['pool_hash']
            with torch.inference_mode():
                q=models[prop](Batch.from_data_list([mol_to_graph(seed)]));q=(q/q.norm(dim=1,keepdim=True)).numpy()[0]
            g=embeddings[prop]@q
            ranked={'M':list(keys[indices[np.lexsort((keys[indices],-m[indices]))[:3]]]),'G':list(keys[indices[np.lexsort((keys[indices],-g[indices]))[:3]]])}
            for repeat in range(1,6):
                number=int.from_bytes(hashlib.sha256(f'42|{prop}|{t["chemblid"]}|{repeat}'.encode()).digest()[:16],'big')
                ranked['R']=list(keys[np.random.default_rng(number).choice(indices,3,replace=False)])
                for group in ['M','G','R']:
                    frozen=expected[(prop,t['chemblid'],group,repeat)];chosen=ranked[group];match=chosen==frozen;checked+=3;exact+=sum(x==y for x,y in zip(chosen,frozen))
                    if not match:
                        assert group=='G',(prop,t['chemblid'],group)
                        ix=[int(np.flatnonzero(keys==s)[0]) for s in frozen];deficit=max(0.,float(np.sort(g[indices])[-3]-min(g[ix])))
                        assert deficit<1e-5,(prop,t['chemblid'],deficit)
                        if repeat==1:near.append(dict(property=prop,chemblid=t['chemblid'],frozen=frozen,reselected=chosen,score_deficit=deficit))
                    for rank,smiles in enumerate(chosen,1):selected.append(dict(property=prop,chemblid=t['chemblid'],group=group,position=repeat,rank=rank,smiles=smiles,frozen_set_exact=match))
    assert checked==2250 and len(selected)==2250
    with (out/'references.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(selected[0]));w.writeheader();w.writerows(selected)
    proof=dict(status='passed_exact' if not near else 'passed_with_float32_near_ties',reference_records=checked,exact_reference_positions=exact,
        Morgan_and_random_exact=True,GIN_all_ordered_sets_exact=not near,near_tie_tasks=near,mode='fresh_library' if a.recompute_library else 'cached_library',
        embedding_max_difference=embedding_diff,checkpoint_sha256={pr:sha(G/f'models/{pr}_best.pth') for pr in ['logp','tpsa']},
        python=sys.version.split()[0],torch=torch.__version__,numpy=np.__version__,API_calls=0,training_jobs=0)
    (out/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
