"""Resumable structural comparisons of all eligible saved candidates and direct baselines."""
import argparse,fcntl,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from rdkit import Chem,RDLogger
from rdkit.Chem import rdFMCS
import pandas as pd
from common import ROOT,read,read_jsonl,sha,now
from analysis_utils import mcs_key,save_json


def protocol():
    return dict(maximizeBonds=False,matchValences=False,ringMatchesRingOnly=False,completeRingsOnly=False,
        matchChiralTag=False,atomCompare='CompareElements',bondCompare='CompareOrder',
        timeouts=[2,10,30],source_code_sha256=sha(__file__))


def compute(job):
    RDLogger.DisableLog('rdApp.*');key,seed,candidate=job;started=time.monotonic()
    a,b=Chem.MolFromSmiles(seed),Chem.MolFromSmiles(candidate);records=[];best=None
    for timeout in [2,10,30]:
        r=rdFMCS.FindMCS([a,b],maximizeBonds=False,timeout=timeout,matchValences=False,
            ringMatchesRingOnly=False,completeRingsOnly=False,matchChiralTag=False,
            atomCompare=rdFMCS.AtomCompare.CompareElements,bondCompare=rdFMCS.BondCompare.CompareOrder)
        records.append(dict(timeout=timeout,canceled=bool(r.canceled),numAtoms=r.numAtoms,numBonds=r.numBonds))
        if best is None or r.numAtoms>best.numAtoms:best=r
        if not r.canceled:best=r;break
    assert best.numAtoms<=min(a.GetNumHeavyAtoms(),b.GetNumHeavyAtoms())
    complete=not records[-1]['canceled'];lower=best.numAtoms/a.GetNumHeavyAtoms()
    out=dict(key=key,seed=seed,canonical=candidate,complete=complete,common_atoms=best.numAtoms,common_bonds=best.numBonds,
        smarts=best.smartsString,seed_atoms=a.GetNumHeavyAtoms(),candidate_atoms=b.GetNumHeavyAtoms(),
        mcs_seed_fraction=lower if complete else None,mcs_lower_seed_fraction=lower,
        mcs_upper_seed_fraction=lower if complete else min(a.GetNumHeavyAtoms(),b.GetNumHeavyAtoms())/a.GetNumHeavyAtoms(),
        attempts=records,elapsed_seconds=round(time.monotonic()-started,4),protocol=protocol())
    save_json(ROOT/'analysis/mcs_cache'/f'{key}.json',out);return out


def collect():
    tasks={r['chemblid']:r['seed_smiles'] for r in read(ROOT/'inputs/tasks25.json')};pairs={}
    for row in pd.read_csv(ROOT/'inputs/direct_baselines150.csv').to_dict('records'):
        seed=tasks[row['chemblid']];pairs[mcs_key(seed,row['canonical'])]=(seed,row['canonical'])
    paths=list((ROOT/'results').glob('*.json'))
    for p in paths:
        result=read(p);seed=tasks[result['chemblid']]
        for c in result.get('candidates',[]):
            if c['eligible']:pairs[mcs_key(seed,c['canonical'])]=(seed,c['canonical'])
    return pairs,len(paths)


def main(watch=False):
    (ROOT/'analysis/mcs_cache').mkdir(exist_ok=True)
    guard=(ROOT/'logs/mcs.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    save_json(ROOT/'analysis/mcs_protocol.json',protocol());active={};seen=set();computed=0
    with ProcessPoolExecutor(max_workers=4) as pool:
        while True:
            pairs,completed_positions=collect()
            for key,(seed,candidate) in pairs.items():
                path=ROOT/'analysis/mcs_cache'/f'{key}.json'
                if key in seen:continue
                if path.exists():assert read(path)['protocol']==protocol();seen.add(key);continue
                active[pool.submit(compute,(key,seed,candidate))]=key;seen.add(key)
            if active:
                done,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
                for future in done:
                    result=future.result();active.pop(future);computed+=1
                    if computed%50==0:print('MCS',computed,'new pairs; pending',len(active),flush=True)
            status=dict(time_utc=now(),completed_positions=completed_positions,known_pairs=len(pairs),pending=len(active),
                cached_pairs=len(list((ROOT/'analysis/mcs_cache').glob('*.json'))),new_API_calls=0)
            save_json(ROOT/'logs/mcs_progress.json',status)
            if not active and (completed_positions==1000 or not watch):break
            if not active:time.sleep(20)
    save_json(ROOT/'analysis/mcs_complete.json',status);print('MCS STATUS',status,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args();main(args.watch)
