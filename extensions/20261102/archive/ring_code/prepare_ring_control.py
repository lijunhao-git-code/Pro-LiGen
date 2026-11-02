"""Freeze the original ring protocol and a same-pool random reference arm."""
from pathlib import Path
import hashlib,json,random,re,shutil,sys
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger,rdBase
from rdkit.Chem import rdFingerprintGenerator
from common import ROOT,PROJECT,sha,now,dump,read,read_jsonl,properties,canon
from service import get_balance,get_models


def main():
    assert not (ROOT/'inputs/generation_locked.json').exists(),'Already prepared; do not alter frozen inputs'
    for name in ['source','inputs','analysis','logs','requests','responses','results']:(ROOT/name).mkdir(exist_ok=True)
    RDLogger.DisableLog('rdApp.*')
    old=PROJECT/'mw_experiments/gin_downstream_20261011_375'
    expected=read(old/'protocol.json');assert expected['created_utc'].startswith('2026-10-11')
    sources={'protocol_original.json':old/'protocol.json','prompts_original75.jsonl':old/'inputs/prompts.jsonl',
             'library.csv':old/'inputs/library.csv','exemplars_original150.csv':old/'inputs/exemplars.csv',
             'targets_original25.csv':old/'inputs/targets.csv','old_GIN_library_embeddings.npy':old/'inputs/old_GIN_library_embeddings.npy',
             'best_gin_rc.pth':old/'inputs/best_gin_rc.pth','encoder_source.json':old/'inputs/encoder_source.json'}
    original_hashes={str(p):sha(p) for p in sources.values()}
    for name,path in sources.items():shutil.copy2(path,ROOT/'source'/name)
    lib=pd.read_csv(ROOT/'source/library.csv');assert len(lib)==26108 and lib.canonical.is_unique
    prompts=read_jsonl(ROOT/'source/prompts_original75.jsonl')
    tasks=pd.read_csv(ROOT/'source/targets_original25.csv').to_dict('records')
    assert len(tasks)==25 and len(prompts)==75
    mp={(e['chemblid'],e['group']):e for e in prompts}
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    fps=[fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in lib.canonical]
    keys=lib.canonical.tolist();librc=lib.RingCount.to_numpy(int)
    references=[];entries=[];pools=[];checks=[];pool_masks={}
    for task in tasks:
        target=task['chemblid'];seed=task['seed_smiles'];sprop=properties(seed)
        mask=(librc==task['target_rc'])&(lib.canonical.to_numpy()!=sprop['canonical'])
        ids=sorted(np.flatnonzero(mask).tolist(),key=lambda i:keys[i]);assert len(ids)>=3
        pool_masks[target]=mask
        # One prespecified random set per task, fixed across all five repeats and rounds,
        # matching the original deterministic backends' reference reuse.
        seedvalue=int.from_bytes(hashlib.sha256(('ring_R_20261102_seed42|'+target).encode()).digest()[:8],'big')
        picked=random.Random(seedvalue).sample(ids,3)
        rrefs=[keys[i] for i in picked]
        mentry=mp[(target,'M')]
        marker=r'(?m)^(  - Exemplar [123]（环数=\d+）: )[^\n]+$'
        assert len(re.findall(marker,mentry['prompt'][3]))==3
        refiter=iter(rrefs)
        rprompt=list(mentry['prompt']);rprompt[3]=re.sub(marker,lambda m:m.group(1)+next(refiter),rprompt[3])
        qp=fpgen.GetFingerprint(Chem.MolFromSmiles(seed))
        tan=np.asarray(DataStructs.BulkTanimotoSimilarity(qp,fps))
        pools.append(dict(chemblid=target,seed_rc=task['seed_rc'],target_rc=task['target_rc'],pool_size=len(ids),
            random_seed=seedvalue,canonical_pool_sha256=hashlib.sha256('\n'.join(keys[i] for i in ids).encode()).hexdigest(),
            pool_multifragment_count=sum('.' in keys[i] for i in ids)))
        for group in ['T','M','G','R']:
            base=dict(mentry,group='R',prompt=rprompt,exemplars=rrefs) if group=='R' else mp[(target,group)]
            base=dict(base)
            assert base['seed_smiles']==seed and base['target_rc']==sprop['RC']-1
            for n in [0,1,2,4]:assert base['prompt'][n]==mentry['prompt'][n]
            if group in ['M','G','R']:
                assert len(base['exemplars'])==len(set(base['exemplars']))==3
                assert re.sub(marker,'EXEMPLAR',base['prompt'][3])==re.sub(marker,'EXEMPLAR',mentry['prompt'][3])
                for rank,smi in enumerate(base['exemplars'],1):
                    index=keys.index(smi);assert mask[index]
                    p=properties(smi);assert p['RC']==task['target_rc'] and p['canonical']!=sprop['canonical']
                    references.append(dict(chemblid=target,group=group,rank=rank,library_id=lib.iloc[index].query_id,
                        smiles=smi,tanimoto=float(tan[index]),**p))
            for position in range(1,6):entries.append(dict(base,position=position,slot_id=f'{target}_{position}_{group}'))
        checks.append(dict(chemblid=target,shared_pool=True,R_only_replaces_three_reference_strings=True,
            original_T_M_G_prompts_identical=True,random_sample_not_tuned=True))
    assert len(entries)==500 and len(references)==225
    targets=[r['chemblid'] for r in tasks]
    rng=np.random.default_rng(42);blocks=[(t,p) for t in targets for p in range(1,6)];rng.shuffle(blocks)
    schedule=[]
    for t,position in blocks:
        for group in rng.permutation(['T','M','G','R']):
            schedule.append(dict(slot_id=f'{t}_{position}_{group}',chemblid=t,position=position,group=str(group),
                order=len(schedule)+1,pilot=(t in targets[:2] and position==1)))
    assert sum(s['pilot'] for s in schedule)==8
    pd.DataFrame(references).to_csv(ROOT/'inputs/references225.csv',index=False)
    pd.DataFrame(pools).to_csv(ROOT/'inputs/reference_pools25.csv',index=False)
    records=pd.DataFrame(references)
    summary=records.groupby('group')[['RC','tanimoto','QED','SA_raw','MW','fragments']].agg(['mean','std','min','median','max'])
    summary.columns=['_'.join(c) for c in summary.columns];summary.reset_index().to_csv(ROOT/'analysis/reference_distribution_summary.csv',index=False)
    with (ROOT/'inputs/entries500.jsonl').open('w',encoding='utf-8') as f:
        for e in entries:f.write(json.dumps(e,ensure_ascii=False)+'\n')
    dump(ROOT/'inputs/schedule500.json',schedule);dump(ROOT/'inputs/tasks25.json',tasks)
    api=dict(expected['api']);api.update(cost_cap_cny=60.,balance_reserve_cny=2.,workers_pilot=4,workers_full=12,
        documented_version_at_preparation='DeepSeek-V4.1-Flash-1017',SDK_retries=0)
    protocol=dict(status='prepared_pending_paid_scope_confirmation',prepared_utc=now(),groups=['T','M','G','R'],
        planned_slots=500,targets=targets,repeats=5,slots_per_group=125,conversation_calls=2500,
        candidate_positions=3000,pilot_slots=8,system_prompt=expected['system_prompt'],api=api,
        original_protocol_sha256=sha(old/'protocol.json'),original_checkpoint_sha256=sha(old/'inputs/best_gin_rc.pth'),
        scientific_scope='Same-ring-condition contemporary replication with same-pool random reference control; do not pool with original October11 results.',
        target='exact original parent-form RC_seed -1',seed_forms='Unmodified original Study-B seed forms, including the original CHEMBL1821 counterion.',
        reference_pool='Exact original RC pool with seed excluded; no extra fragment/QED/SA/Tanimoto filter, identical for M/G/R.',
        M_G='Reuse exactly the original 75+75 ranked references and original checkpoint; no retraining.',
        R='Uniform three distinct library entries from canonical-sorted common pool; seed42+task-derived fixed seed, same set for five repeats and both rounds.',
        generation='Original five prompts, original system prompt; only R example strings differ from M. Independent contexts per position; two proposal rounds x3.',
        eligibility='Unchanged original code: valid nonempty single-component, exact RC, no seed or supplied-example canonical copy.',
        selection='Eligible union; last canonical occurrence; QED descending, SA_raw ascending, canonical ascending.',
        primary=dict(contrasts=['M-R','G-R'],endpoints=['S','U'],family='Holm across4 primary tests',unit='25 target paired means',bootstrap=10000,sign_flips=100000,seed=42),
        secondary=dict(contrasts=['R-T','G-M'],endpoints=['S','U'],family='separate Holm across4 secondary tests'),
        descriptive=['candidate quality','first/random-unique-eligible/maximum-QED selection','QED','SA','Tanimoto','MW','uniqueness','library/training exact overlap'],
        no_docking_in_initial_R_control=True,no_manuscript_edits=True,no_publication=True,
        pause_policy='Before-API budget and live balance gates, durable successful responses, no quality retries; bounded technical retries only.',
        pilot_policy='Eight predetermined positions; inspect response parsing, property computations, feedback, noncopy and selector before bulk; scientific failure does not trigger protocol tuning.')
    dump(ROOT/'protocol.json',protocol)
    dump(ROOT/'AUTHORIZATION.json',dict(approved=False,planned_slots=500,cost_cap_cny=60.,reason='Awaiting confirmation because new R alone is not contemporaneous with original375; no paid calls allowed yet.'))
    probe=dict(balance=get_balance(),models=get_models(),time_utc=now(),paid_calls=0,credentials_recorded=False)
    dump(ROOT/'inputs/api_probe.json',probe)
    assert all(sha(p)==h for p,h in original_hashes.items())
    dump(ROOT/'inputs/protected_sources.json',original_hashes)
    dump(ROOT/'inputs/preparation_checks.json',dict(status='passed',tasks=25,entries=500,reference_records=225,pilot_slots=8,
        common_pools_identical=True,R_uniform_sample_saved=True,original_prompts_and_M_G_refs_unchanged=True,
        original_sources_unchanged=True,per_target=checks,rdkit=rdBase.rdkitVersion,new_paid_calls=0))
    # Lock after no-API runner self-tests and final preparation review.
    print(json.dumps(dict(status='prepared',planned_slots=500,reference_records=225,pilot_slots=8,
        balance_cny=probe['balance']['balance_cny'],new_paid_calls=0,pending_scope_confirmation=True)))


if __name__=='__main__':main()
