"""Freeze attribute references, all1,000 five-turn inputs and randomized schedule."""
import hashlib,json,os,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8');os.environ.setdefault('OMP_NUM_THREADS','4')
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from torch_geometric.data import Batch,Data
from rdkit import RDLogger
from common import ROOT,PROPERTIES,sha,dump,read,read_jsonl,properties,now
from gin_model_original import GINEncoder,mol_to_graph


def embed(model,graphs,device):
    result=[]
    with torch.inference_mode():
        for offset in range(0,len(graphs),128):
            raw=model(Batch.from_data_list(graphs[offset:offset+128]).to(device))
            assert torch.isfinite(raw).all() and (raw.norm(dim=1)>1e-12).all()
            result.append(F.normalize(raw,dim=1).cpu().numpy())
    return np.concatenate(result)


def top3(scores,eligible,keys,k=3):
    idx=np.flatnonzero(eligible&np.isfinite(scores))
    assert len(idx)>=k
    cutoff=np.partition(scores[idx],-k)[-k];idx=idx[scores[idx]>=cutoff]
    return idx[np.lexsort((keys[idx],-scores[idx]))[:k]]


def design_prompt(entry):
    prop=entry['property'];info=PROPERTIES[prop]
    refs=''
    if entry['exemplars']:
        refs='- 满足相同属性目标区间的参考分子（共3个）：\n'+'\n'.join(
            f'  - Exemplar {i}（{info["label"]}={properties(s)[prop]:.8f}）：{s}' for i,s in enumerate(entry['exemplars'],1))+'\n'
        approach='分析参考分子相对于种子的结构改造方式，结合种子及靶点信息进行设计。'
    else:approach='分析种子的结构改造方式，结合种子及靶点信息进行设计。'
    return f'''你是一位经验丰富的药物化学专家，擅长启发式分子设计。
【任务】
1. {approach}
2. 生成3个全新的、化学上有效的SMILES，按当前单属性条件进行温和改造，并尽量保留种子的关键结构。
3. 只设计单个连通且含碳的分子，不要直接输出种子或输入中提供的参考分子。
【输入信息】
- 靶点UniProt ID：{entry['uniprotid']}
- 靶点ChEMBL ID：{entry['chemblid']}
- 种子SMILES：{entry['seed_smiles']}
- 种子{info['label']}：{entry['seed_property']:.8f}
{refs}【目标属性】
- 当前唯一性质硬条件：{info['label']}相对种子降低{abs(info['delta']):g}，允许误差±{info['tolerance']:g}。
- 因此每个输出的RDKit实算{info['label']}应位于[{entry['target_lower']:.8f}, {entry['target_upper']:.8f}]（含端点）。
- 不额外要求环数、分子量或另一属性达到某个数值。
【输出格式】
## Structural Analysis
用1至2句话说明改造策略。
## Generated Molecules
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>
仅输出3个带标签候选。不要编造QED、SA、Vina或实验活性，也不要声称已实际对接。'''


def optimize_prompt(entry):
    info=PROPERTIES[entry['property']]
    return f'''依据前文靶点、文献、种子以及程序实算结果，进行一次分子优化。
唯一性质硬条件：{info['label']}应位于[{entry['target_lower']:.8f}, {entry['target_upper']:.8f}]。
生成单连通含碳分子，不复制种子或输入范例；在达标前提下提高类药性并注意合成可行性和结构保留。
程序会合并两轮各3个候选：按实际QED从高到低，完全同分时按SA原始分数从低到高，再按canonical SMILES排序。
只提出3个候选，不声称自行执行RDKit、QSAR或对接，不编造性质和结合结果。
简短解释后只输出：
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>'''


def main():
    if (ROOT/'inputs/generation_locked.json').exists():raise RuntimeError('Generation already frozen')
    assert read(ROOT/'training/complete.json')['status']=='complete'
    RDLogger.DisableLog('rdApp.*');torch.set_num_threads(4)
    tasks=read(ROOT/'inputs/tasks25.json');lib=pd.read_csv(ROOT/'inputs/library.csv')
    bits=np.load(ROOT/'inputs/morgan_bits.npy');counts=bits.sum(axis=1);keys=lib.canonical.to_numpy(str)
    original=[r for r in read_jsonl(ROOT/'source/prompts_B.jsonl') if r['group']=='T']
    prior={r['chemblid']:r['prompt'] for r in original};assert len(prior)==25
    graphs=[Data(**r) for r in torch.load(ROOT/'training/graphs.pt',map_location='cpu',weights_only=True)]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embedding={};models={};diagnostics=[]
    for prop in PROPERTIES:
        model=GINEncoder(84,256,256,5).to(device)
        model.load_state_dict(torch.load(ROOT/f'models/{prop}_best.pth',map_location=device,weights_only=True),strict=True);model.eval()
        z=embed(model,graphs,device);assert np.std(z,axis=0).mean()>1e-8
        np.save(ROOT/f'inputs/{prop}_library_embeddings.npy',z);embedding[prop]=z;models[prop]=model
        diagnostics.append(dict(property=prop,mean_embedding_std=float(np.std(z,axis=0).mean()),checkpoint_sha256=sha(ROOT/f'models/{prop}_best.pth')))
        print('Embedded library',prop,len(z),flush=True)
    # Work with the original fingerprint bits, not GIN embeddings, for M ranking.
    from rdkit import Chem,DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    entries=[];reference=[];pools=[];checks=[];baselines=[];baseline_candidates=[]
    for task in tasks:
        seed=task['seed_smiles'];arr=np.zeros(2048,dtype=np.float32)
        DataStructs.ConvertToNumpyArray(fpgen.GetFingerprint(Chem.MolFromSmiles(seed)),arr)
        m_scores=np.rint(bits@arr)/np.sqrt(np.maximum(counts*arr.sum(),1e-10))
        base=list(prior[task['chemblid']][:3])
        base[0]=base[0].replace('环数按 RDKit RingCount 定义，不是 SMILES 数字的个数。','目标性质由外部RDKit程序定义和计算。')
        assert len(base)==3 and all('恰好比种子配体少' not in p for p in base)
        for prop,settings in PROPERTIES.items():
            center=task[prop]+settings['delta'];tol=settings['tolerance']
            eligible=(abs(lib[prop].to_numpy()-center)<=tol+1e-8)&(keys!=seed)
            qgraph=mol_to_graph(seed);qa=embed(models[prop],[qgraph],device)[0];qb=embed(models[prop],[qgraph],device)[0]
            assert np.allclose(qa,qb,atol=1e-6,rtol=0)
            g_scores=embedding[prop]@qa;repeat_scores=embedding[prop]@qb
            mg={'M':top3(m_scores,eligible,keys),'G':top3(g_scores,eligible,keys)}
            assert np.array_equal(mg['G'],top3(repeat_scores,eligible,keys))
            pool_indices=np.flatnonzero(eligible)
            ranked=top3(m_scores,eligible,keys,6)
            for method,k in [('LIB_NN1',1),('LIB_QED3',3),('LIB_QED6',6)]:
                options=[]
                for rank,idx in enumerate(ranked[:k],1):
                    p=properties(str(keys[idx]));shared=float(np.dot(arr,bits[idx]))
                    record=dict(property=prop,chemblid=task['chemblid'],group=method,position=1,rank=rank,
                        canonical=p['canonical'],library_id=lib.iloc[idx].library_id,QED=p['QED'],SA_raw=p['SA_raw'],
                        property_value=p[prop],target_error=p[prop]-center,delta_property=p[prop]-task[prop],
                        tanimoto=float(shared/(arr.sum()+counts[idx]-shared)),cosine=float(m_scores[idx]),
                        seed_smiles=seed,seed_property=task[prop],target_center=center,tolerance=tol,
                        MW=p['MW'],HeavyAtoms=p['HeavyAtoms'],RC=p['RC'],copy_exemplar_allowed=True)
                    options.append(record);baseline_candidates.append(record)
                chosen=min(options,key=lambda r:(-r['QED'],r['SA_raw'],r['canonical']))
                baselines.append(dict(chosen,slot_id=f'{prop}_{task["chemblid"]}_{method}'))
            pools.append(dict(property=prop,chemblid=task['chemblid'],seed_property=task[prop],center=center,tolerance=tol,
                pool_size=int(eligible.sum()),m_g_overlap=len(set(mg['M'])&set(mg['G'])),
                pool_hash=__import__('hashlib').sha256('\n'.join(sorted(keys[eligible])).encode()).hexdigest()))
            for repeat in range(1,6):
                random_seed=int.from_bytes(hashlib.sha256(f'42|{prop}|{task["chemblid"]}|{repeat}'.encode()).digest()[:16],'big')
                ridx=np.random.default_rng(random_seed).choice(pool_indices,size=3,replace=False)
                for group in ['T','M','G','R']:
                    chosen=[] if group=='T' else (ridx if group=='R' else mg[group])
                    exemplars=[str(keys[i]) for i in chosen]
                    entry=dict(property=prop,chemblid=task['chemblid'],uniprotid=task['uniprotid'],group=group,position=repeat,
                        slot_id=f'{prop}_{task["chemblid"]}_{repeat}_{group}',seed_smiles=seed,seed_property=task[prop],
                        target_center=center,tolerance=tol,target_lower=center-tol,target_upper=center+tol,
                        exemplars=exemplars,random_seed=str(random_seed) if group=='R' else None)
                    entry['prompt']=base+[design_prompt(entry),optimize_prompt(entry)]
                    assert len(entry['prompt'])==5
                    entries.append(entry)
                    for rank,index in enumerate(chosen,1):
                        p=properties(str(keys[index]));assert abs(p[prop]-center)<=tol+1e-8 and p['canonical']!=seed
                        qset=np.flatnonzero(arr);shared=float(np.dot(arr,bits[index]));tanimoto=shared/(arr.sum()+counts[index]-shared)
                        reference.append(dict(slot_id=entry['slot_id'],property=prop,chemblid=task['chemblid'],group=group,position=repeat,
                            rank=rank,library_id=lib.iloc[index].library_id,smiles=str(keys[index]),property_value=p[prop],
                            target_error=p[prop]-center,tanimoto=float(tanimoto),QED=p['QED'],SA_raw=p['SA_raw'],MW=p['MW'],RC=p['RC']))
            checks.append(dict(property=prop,chemblid=task['chemblid'],repeat_seed_embedding_ranking_identical=True))
    assert len(entries)==1000 and len({r['slot_id'] for r in entries})==1000 and len(reference)==2250
    index={e['slot_id']:e for e in entries};rng=np.random.default_rng(42)
    blocks=[(prop,t['chemblid'],repeat) for prop in PROPERTIES for t in tasks for repeat in range(1,6)]
    schedule=[]
    for bi in rng.permutation(len(blocks)):
        prop,target,repeat=blocks[bi]
        for group in rng.permutation(['T','M','G','R']):
            sid=f'{prop}_{target}_{repeat}_{group}'
            schedule.append(dict(slot_id=sid,property=prop,chemblid=target,position=repeat,group=str(group),order=len(schedule)+1))
    pilot_set={f'{prop}_{t["chemblid"]}_1_{group}' for prop in PROPERTIES for t in tasks[:2] for group in ['T','M','G','R']}
    assert len(pilot_set)==16
    for row in schedule:row['pilot']=row['slot_id'] in pilot_set
    with (ROOT/'inputs/entries1000.jsonl').open('w',encoding='utf-8') as stream:
        for row in entries:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    pd.DataFrame(reference).to_csv(ROOT/'inputs/references2250.csv',index=False)
    pd.DataFrame(pools).to_csv(ROOT/'inputs/reference_pools50.csv',index=False)
    pd.DataFrame(baselines).to_csv(ROOT/'inputs/direct_baselines150.csv',index=False)
    pd.DataFrame(baseline_candidates).to_csv(ROOT/'inputs/direct_baseline_candidates.csv',index=False)
    protocol_reference=read(ROOT/'source/protocol_v2.json')
    grids={r['chemblid']:r for r in read(ROOT/'source/receptor_grids25.json')}
    receptors={}
    for target in protocol_reference['docking']['target_ids']:
        path=ROOT/'docking/receptors'/target/'receptor.pdbqt';r=grids[target]
        assert path.is_file() and sha(path)==r['pdbqt_sha256']
        receptors[target]=dict(chemblid=target,path=path.relative_to(ROOT).as_posix(),sha256=sha(path),
            box_center=r['box_center'],box_size=r['box_size'],pdb_id=r['pdb_id'])
    assert len(receptors)==11
    dump(ROOT/'inputs/docking_receptors11.json',receptors)
    from common import sa_scorer
    from rdkit import RDConfig
    scorer_path=__import__('pathlib').Path(RDConfig.RDContribDir)/'SA_Score'
    dump(ROOT/'inputs/standard_SA_provenance.json',{n:sha(scorer_path/n) for n in ['sascorer.py','fpscores.pkl.gz']})
    dump(ROOT/'inputs/schedule1000.json',schedule)
    protocol=read(ROOT/'source/protocol_v2.json');protocol.update(status='frozen_before_first_paid_generation',locked_utc=now(),
        actual_runtime_models=diagnostics,pilot_review_required_before_full_concurrency=True,
        pilot_review='Check task interpretation, valid actual structures, descriptor windows, factual program feedback, selection and resume. No requirement that GIN wins. Systematic problems must be resolved with explicit protocol-version records.',
        balance=dict(reserve_cny=2.,check_before_paid_requests=True,live_check_interval_seconds=15,live_check_every_calls=8,
                     insufficient='Stop submitting, drain in-flight replies to durable storage, preserve per-call/slot state, resume same slot and cached successful turns after top-up.'))
    protocol['structural_sensitivity']=dict(metrics=['Morgan parent Tanimoto','MCS seed-heavy-atom coverage'],
        thresholds=[round(i/10,1) for i in range(10)],policies=['filter original final','reselect from retained eligible candidates'],
        interpretation='Descriptive saved-candidate sensitivity; not additional paid generation or a hard constraint in the primary experiment.',
        mcs='Maximum atoms, elements/bond order, ring-to-chain allowed, no chirality/charge match; timeouts2/10/30s with completion flags and lower/upper bounds.')
    protocol['pricing_provenance']=dict(checked_client_date='2026-10-28',source='https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
        peak_cny_per_million=dict(cache_hit=.04,cache_miss=2.,output=8.),offpeak_cny_per_million=dict(cache_hit=.02,cache_miss=1.,output=4.),
        prices_are_estimates=True,balance_is_live=True)
    dump(ROOT/'protocol.json',protocol)
    freeze_paths=[ROOT/'protocol.json',ROOT/'inputs/entries1000.jsonl',ROOT/'inputs/references2250.csv',ROOT/'inputs/reference_pools50.csv',
        ROOT/'inputs/schedule1000.json',ROOT/'inputs/library.csv',ROOT/'inputs/tasks25.json',ROOT/'inputs/api_probe.json',
        ROOT/'inputs/direct_baselines150.csv',ROOT/'inputs/direct_baseline_candidates.csv',ROOT/'inputs/docking_receptors11.json',
        ROOT/'inputs/standard_SA_provenance.json',ROOT/'inputs/runner_self_tests.json']
    freeze_paths+=list((ROOT/'docking/receptors').glob('*/receptor.pdbqt'))
    freeze_paths+=[ROOT/'code'/name for name in ['common.py','gin_model_original.py','prepare_generation.py','run_generation.py','service.py']]
    freeze_paths+=[ROOT/f'models/{p}_best.pth' for p in PROPERTIES]
    dump(ROOT/'inputs/generation_locked.json',dict(time_utc=now(),files={p.relative_to(ROOT).as_posix():sha(p) for p in freeze_paths},
        rule='No scientific changes after first paid generation. Preserve each technical code patch and reverify frozen scientific inputs.'))
    dump(ROOT/'inputs/generation_precheck.json',dict(status='passed',entries=1000,references=2250,pools=50,pilot=16,seed_embedding_checks=checks,
        new_generation_calls=0,all_group_common_source_prompts=True,all_R_examples_fixed_per_run=True,all_pools_common=True))
    print('GENERATION INPUTS LOCKED:1000 positions,16 pilot,5 calls per position, no paid calls yet.',flush=True)


if __name__=='__main__':main()
