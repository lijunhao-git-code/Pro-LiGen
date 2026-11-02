"""Two fixed property adaptations from the same frozen initializer, no LLM feedback."""
import csv,json,os,random,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('OMP_NUM_THREADS','4');os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import numpy as np,pandas as pd,torch
from torch_geometric.data import Batch,Data
from common import ROOT,sha,dump,read,now
from gin_model_original import GINEncoder,infonce_loss


def main():
    assert read(ROOT/'inputs/data_prepared.json')['status']=='passed'
    frozen=read(ROOT/'training/input_hashes.json')
    assert all(sha(ROOT/name)==v for name,v in frozen.items())
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cache=torch.load(ROOT/'training/graphs.pt',map_location='cpu',weights_only=True)
    graphs=[Data(**x) for x in cache];del cache
    print('Training device',device,'graphs',len(graphs),flush=True)
    for prop in ['logp','tpsa']:
        status_path=ROOT/f'training/{prop}_status.json'
        if status_path.exists() and read(status_path).get('status')=='complete':continue
        if (ROOT/f'models/{prop}_best.pth').exists():raise RuntimeError('Unfinished training exists; preserve and review before restarting')
        random.seed(42);np.random.seed(42);torch.manual_seed(42)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(42)
        matrices={split:pd.read_csv(ROOT/f'training/{prop}_{split}_triplets.csv')[['anchor_index','positive_index','negative_index']].to_numpy(int) for split in ['train','val']}
        model=GINEncoder(84,256,256,5).to(device)
        model.load_state_dict(torch.load(ROOT/'models/initializer.pth',map_location=device,weights_only=True),strict=True)
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5)
        best=float('inf');patience=0;started=time.monotonic();rng=np.random.default_rng(42)
        log=ROOT/f'training/{prop}_epochs.csv'
        with log.open('x',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=['epoch','train_loss','val_loss','elapsed_seconds']);writer.writeheader();stream.flush()
            for epoch in range(1,51):
                losses={}
                for split in ['train','val']:
                    model.train(split=='train');matrix=matrices[split]
                    order=rng.permutation(len(matrix)) if split=='train' else np.arange(len(matrix))
                    total=0.;batches=0
                    context=torch.enable_grad() if split=='train' else torch.inference_mode()
                    with context:
                        for offset in range(0,len(order),32):
                            indices=matrix[order[offset:offset+32]]
                            a,p,n=[Batch.from_data_list([graphs[int(i)] for i in indices[:,j]]).to(device) for j in range(3)]
                            loss=infonce_loss(model(a),model(p),model(n),.1)
                            if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
                            if split=='train':optimizer.zero_grad();loss.backward();optimizer.step()
                            total+=float(loss.item());batches+=1
                    losses[split]=total/max(1,batches)
                row=dict(epoch=epoch,train_loss=losses['train'],val_loss=losses['val'],elapsed_seconds=round(time.monotonic()-started,3))
                writer.writerow(row);stream.flush();os.fsync(stream.fileno())
                if losses['val']<best:
                    best=losses['val'];patience=0
                    temp=ROOT/f'models/{prop}_best.tmp';torch.save(model.state_dict(),temp);os.replace(temp,ROOT/f'models/{prop}_best.pth')
                    best_epoch=epoch
                else:patience+=1
                dump(status_path,dict(status='running',property=prop,epoch=epoch,best_epoch=best_epoch,best_val=best,patience=patience,
                    rows=matrices['train'].shape[0],updated_utc=now(),elapsed_seconds=row['elapsed_seconds'],device=str(device),new_API_calls=0))
                print(prop,json.dumps(row),'best_epoch',best_epoch,'patience',patience,flush=True)
                if patience>=10:break
        dump(status_path,dict(status='complete',property=prop,epochs=epoch,best_epoch=best_epoch,best_val=best,
            checkpoint_sha256=sha(ROOT/f'models/{prop}_best.pth'),initializer_sha256=sha(ROOT/'models/initializer.pth'),
            updated_utc=now(),elapsed_seconds=round(time.monotonic()-started,3),device=str(device),new_API_calls=0))
    dump(ROOT/'training/complete.json',dict(status='complete',properties=['logp','tpsa'],completed_utc=now(),new_API_calls=0))
    print('BOTH PROPERTY MODELS COMPLETE',flush=True)


if __name__=='__main__':main()
