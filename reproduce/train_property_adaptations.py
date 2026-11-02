"""Validate or rerun the released property adaptations in a new directory.
Default action validates data and one forward loss without an optimizer step.
Explicit --run-training prepares graph caches and calls the archived trainer.
"""
from pathlib import Path
import argparse,csv,hashlib,importlib.util,json,shutil,sys
import numpy as np,torch
from torch_geometric.data import Batch
ROOT=Path(__file__).resolve().parents[1];G=ROOT/'extensions/20261102/gin'
sys.path.insert(0,str(G/'code'))
from gin_model_original import mol_to_graph,GINEncoder,infonce_loss
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--run-training',action='store_true');a=p.parse_args()
    out=Path(a.out).resolve();assert not out.exists();out.mkdir(parents=True)
    torch.set_num_threads(4);lib=rows(G/'inputs/library.csv');assert len(lib)==24693
    counts={}
    for prop in ['logp','tpsa']:
        counts[prop]={}
        for split in ['train','val']:
            part=rows(G/f'training/{prop}_{split}_triplets.csv');counts[prop][split]=len(part)
            for r in part:
                for label in ['anchor','positive','negative']:
                    source=lib[int(r[label+'_index'])];assert source['canonical']==r[label+'_smiles'] and source['split']==split
                    assert abs(float(source[prop])-float(r[label+'_property']))<1e-10
                assert float(r['positive_property'])<float(r['negative_property'])
    assert counts=={'logp':{'train':17334,'val':3650},'tpsa':{'train':16187,'val':3426}}
    model=GINEncoder(84,256,256,5).cpu();model.load_state_dict(torch.load(G/'models/initializer.pth',map_location='cpu',weights_only=True));model.eval()
    sample=rows(G/'training/logp_train_triplets.csv')[:3]
    with torch.inference_mode():
        batches=[Batch.from_data_list([mol_to_graph(r[k+'_smiles']) for r in sample]) for k in ['anchor','positive','negative']]
        loss=float(infonce_loss(*[model(b) for b in batches],.1));assert np.isfinite(loss)
    if a.run_training:
        for folder in ['inputs','training','models']:(out/folder).mkdir()
        for name in ['library.csv','tasks25.json','data_prepared.json']:shutil.copy2(G/'inputs'/name,out/'inputs'/name)
        for path in (G/'training').glob('*_triplets.csv'):shutil.copy2(path,out/'training'/path.name)
        shutil.copy2(G/'models/initializer.pth',out/'models/initializer.pth')
        graphs=[]
        for r in lib:
            g=mol_to_graph(r['canonical']);graphs.append(dict(x=g.x,edge_index=g.edge_index,edge_attr=g.edge_attr))
        torch.save(graphs,out/'training/graphs.pt')
        files=list((out/'training').glob('*'))+[out/'inputs/library.csv',out/'inputs/tasks25.json',out/'models/initializer.pth']
        (out/'training/input_hashes.json').write_text(json.dumps({f.relative_to(out).as_posix():sha(f) for f in files},indent=2),encoding='utf-8')
        spec=importlib.util.spec_from_file_location('archived_trainer',G/'code/train_models.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.ROOT=out;module.main()
    proof=dict(status='passed',triplet_counts=counts,example_loss=loss,new_training_executed=a.run_training,optimizer_steps_in_validation=0,API_calls=0)
    (out/'setup_verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
