"""Standardize labels and build property-consistent nearest-neighbor triplets."""
import importlib.util,json,os,shutil,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np,pandas as pd,torch
from rdkit import Chem,RDLogger,rdBase
from rdkit.Chem import Descriptors,rdFingerprintGenerator
from common import ROOT,PROJECT,PROPERTIES,sha,dump,read,seed_component,properties,now


def main():
    started=time.monotonic()
    if (ROOT/'inputs/data_prepared.json').exists():raise RuntimeError('Data already prepared')
    for d in ['inputs','training','models','logs','requests','responses','results','analysis','figures','reproduce']:(ROOT/d).mkdir(exist_ok=True)
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    original=PROJECT/'mw_experiments'
    init=original/'models/gin_pretrained.pth'
    assert sha(init)=='ad65679dc6b9a9b63fe2f7efd97fb1fd115b2b92c14303d6c492807e725266fe'
    shutil.copy2(init,ROOT/'models/initializer.pth')
    shutil.copy2(original/'train_gin.py',ROOT/'code/gin_model_original.py')
    spec=importlib.util.spec_from_file_location('source_gin',ROOT/'code/gin_model_original.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    tasks=[]
    for source in read(ROOT/'source/tasks25.json'):
        used=seed_component(source['seed_smiles']);props=properties(used)
        tasks.append(dict(chemblid=source['chemblid'],uniprotid=source['uniprotid'],seed_source=source['seed_smiles'],
            seed_smiles=used,seed_form_changed=used!=source['seed_canonical'],**props))
    assert len(tasks)==25
    dump(ROOT/'inputs/tasks25.json',tasks)
    seeds={x['seed_smiles'] for x in tasks}
    source=pd.read_csv(ROOT/'source/library_original.csv')
    rows=[];fps=[];graphs=[];fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    for i,row in source.iterrows():
        mol=Chem.MolFromSmiles(row.canonical)
        if mol is None or len(Chem.GetMolFrags(mol))!=1 or not any(a.GetAtomicNum()==6 for a in mol.GetAtoms()):continue
        props=properties(row.canonical)
        assert props['canonical']==row.canonical
        assert abs(props['logp']-row.Crippen_logP)<1e-8 and abs(props['tpsa']-row.TPSA)<1e-8
        index=len(rows);record=dict(index=index,library_id=row.query_id,split=row.split,**props)
        rows.append(record)
        arr=np.zeros(2048,dtype=np.float32);from rdkit import DataStructs
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(mol),arr);fps.append(arr)
        graph=module.mol_to_graph(row.canonical)
        assert graph is not None and graph.x.shape[1]==84
        graphs.append(dict(x=graph.x,edge_index=graph.edge_index,edge_attr=graph.edge_attr))
        if len(rows)%5000==0:print('Prepared library molecules',len(rows),flush=True)
    lib=pd.DataFrame(rows);assert len(lib)==24693 and lib.canonical.is_unique
    lib.to_csv(ROOT/'inputs/library.csv',index=False)
    bits=np.stack(fps);np.save(ROOT/'inputs/morgan_bits.npy',bits)
    torch.save(graphs,ROOT/'training/graphs.pt')
    print('Labels and cached graphs prepared; building split-specific nearest10 triplets.',flush=True)
    neighbor_tables={};skip_seeds=0
    for split in ['train','val']:
        part=lib[(lib.split==split)&~lib.canonical.isin(seeds)]
        indices=part['index'].to_numpy(int);skip_seeds+=int(((lib.split==split)&lib.canonical.isin(seeds)).sum())
        count=np.sum(bits[indices],axis=1);assert (count>0).all()
        neighbors=[];keys=part.canonical.to_numpy(str)
        for offset in range(0,len(indices),128):
            anchor_global=indices[offset:offset+128]
            shared=np.rint(bits[anchor_global]@bits[indices].T)
            cosine=shared/np.sqrt(count[offset:offset+len(anchor_global),None]*count[None,:])
            for local,row in enumerate(cosine):
                row[offset+local]=-np.inf
                cutoff=np.partition(row,-10)[-10];near=np.flatnonzero(row>=cutoff)
                selected=near[np.lexsort((keys[near],-row[near]))[:10]]
                assert len(selected)==10
                neighbors.append((int(indices[offset+local]),indices[selected]))
            if offset%2048==0:print('Neighbors',split,offset,'/',len(indices),flush=True)
        neighbor_tables[split]=neighbors
    counts={}
    for prop in PROPERTIES:
        values=lib[prop].to_numpy(float);counts[prop]={}
        for split,neighbors in neighbor_tables.items():
            triplets=[]
            for anchor,near in neighbors:
                vals=values[near];positive=int(near[np.argmin(vals)]);negative=int(near[np.argmax(vals)])
                if positive==negative or values[negative]-values[positive]<=1e-8:continue
                triplets.append(dict(anchor_index=anchor,positive_index=positive,negative_index=negative,
                    anchor_smiles=lib.iloc[anchor].canonical,positive_smiles=lib.iloc[positive].canonical,negative_smiles=lib.iloc[negative].canonical,
                    anchor_property=float(values[anchor]),positive_property=float(values[positive]),negative_property=float(values[negative])))
            frame=pd.DataFrame(triplets)
            assert frame.positive_property.lt(frame.negative_property).all()
            assert set(lib.iloc[frame[['anchor_index','positive_index','negative_index']].to_numpy().ravel()].split)=={split}
            frame.to_csv(ROOT/f'training/{prop}_{split}_triplets.csv',index=False)
            counts[prop][split]=len(frame)
    info=dict(status='passed',created_utc=now(),library=len(lib),triplets=counts,task_seeds_excluded_from_finetuning=skip_seeds,
        seed_form_changed=[x['chemblid'] for x in tasks if x['seed_form_changed']],rdkit=rdBase.rdkitVersion,
        initializer_sha256=sha(ROOT/'models/initializer.pth'),model_source_sha256=sha(ROOT/'code/gin_model_original.py'),
        source_hashes={p.name:sha(p) for p in (ROOT/'source').iterdir() if p.is_file()},
        source_files_unchanged=True,new_API_calls=0,elapsed_seconds=round(time.monotonic()-started,3))
    dump(ROOT/'inputs/data_prepared.json',info)
    dump(ROOT/'training/input_hashes.json',{p.relative_to(ROOT).as_posix():sha(p) for p in list((ROOT/'training').glob('*'))+[ROOT/'inputs/library.csv',ROOT/'inputs/tasks25.json',ROOT/'models/initializer.pth'] if p.is_file()})
    print(json.dumps(info),flush=True)


if __name__=='__main__':main()
