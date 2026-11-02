"""Isolated, resumable Vina jobs on the11 prevalidated receptors; no model/API calls."""
import argparse,fcntl,hashlib,json,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import pandas as pd
from common import ROOT,read,dump,sha,now


def protocol():
    original=read(ROOT/'protocol.json')['docking']['protocol']
    return dict(**original,source_code_sha256=sha(Path(__file__)),standardized_single_component=True)


def collect(pilot_only=False):
    recs=read(ROOT/'inputs/docking_receptors11.json');tasks=read(ROOT/'inputs/tasks25.json')
    schedule=read(ROOT/'inputs/schedule1000.json')
    if pilot_only:schedule=[s for s in schedule if s['pilot']];tasks=tasks[:2]
    observations=[];jobs={};phash=hashlib.sha256(json.dumps(protocol(),sort_keys=True).encode()).hexdigest()
    for task in tasks:
        observations.append(dict(observation_id='seed_'+task['chemblid'],kind='seed',property='shared',group='SEED',
            chemblid=task['chemblid'],canonical=task['seed_smiles'],generation_status='seed_reference'))
    if not pilot_only:
        for row in pd.read_csv(ROOT/'inputs/direct_baselines150.csv').to_dict('records'):
            observations.append(dict(observation_id='direct_'+row['slot_id'],kind='direct_baseline',property=row['property'],group=row['group'],
                chemblid=row['chemblid'],canonical=row['canonical'],generation_status='deterministic_library_output'))
    for slot in schedule:
        path=ROOT/'results'/f'{slot["slot_id"]}.json';result=read(path) if path.exists() else None
        observations.append(dict(observation_id='generated_'+slot['slot_id'],kind='generated',property=slot['property'],group=slot['group'],
            chemblid=slot['chemblid'],slot_id=slot['slot_id'],position=slot['position'],canonical=result.get('canonical') if result else None,
            generation_status=result['status'] if result else 'pending_generation'))
    for row in observations:
        target=row['chemblid'];row['in_primary_docking_subset']=target in recs
        if target not in recs:row.update(docking_status='excluded_receptor_subset',job_id=None);continue
        if not row['canonical']:
            row.update(docking_status='pending_generation' if row['generation_status']=='pending_generation' else 'no_generated_output',job_id=None);continue
        receptor=recs[target];assert sha(ROOT/receptor['path'])==receptor['sha256']
        key=json.dumps([target,row['canonical'],receptor['sha256'],receptor['box_center'],receptor['box_size'],phash],sort_keys=True)
        job=hashlib.sha256(key.encode()).hexdigest()[:24];row.update(docking_status='pending_docking',job_id=job)
        jobs[job]=dict(job_id=job,chemblid=target,canonical=row['canonical'],receptor=receptor,
                      protocol=protocol(),protocol_sha256=phash)
    return observations,jobs


def worker(job_id):
    from rdkit import Chem,RDLogger
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation,PDBQTWriterLegacy
    from vina import Vina
    RDLogger.DisableLog('rdApp.*')
    folder=ROOT/'docking/jobs'/job_id;task=read(folder/'task.json');started=time.monotonic()
    result=dict(job_id=job_id,chemblid=task['chemblid'],canonical=task['canonical'],status='technical_failure',vina=None,
                started_utc=now(),protocol_sha256=task['protocol_sha256'])
    try:
        assert task['protocol']==protocol()
        receptor=task['receptor'];assert sha(ROOT/receptor['path'])==receptor['sha256']
        mol=Chem.AddHs(Chem.MolFromSmiles(task['canonical']))
        if AllChem.EmbedMolecule(mol,randomSeed=42)!=0:
            if AllChem.EmbedMolecule(mol,randomSeed=42,useRandomCoords=True)!=0:raise ValueError('3D embedding failed twice')
        result['mmff_status']=int(AllChem.MMFFOptimizeMolecule(mol,maxIters=500))
        with Chem.SDWriter(str(folder/'prepared_ligand.sdf')) as writer:writer.write(mol)
        setup=MoleculePreparation().prepare(mol)[0];text,ok,error=PDBQTWriterLegacy().write_string(setup)
        if not ok:raise ValueError('PDBQT preparation failed')
        (folder/'ligand.pdbqt').write_text(text)
        vina=Vina(sf_name='vina',cpu=1,seed=42,verbosity=0)
        vina.set_receptor(str(ROOT/receptor['path']));vina.set_ligand_from_file(str(folder/'ligand.pdbqt'))
        vina.compute_vina_maps(center=receptor['box_center'],box_size=receptor['box_size'])
        vina.dock(exhaustiveness=16,n_poses=10);energies=vina.energies(n_poses=10)
        assert len(energies)>0
        vina.write_poses(str(folder/'poses.pdbqt'),n_poses=10,overwrite=True)
        result.update(status='ok',vina=float(energies[0,0]),energies=energies.tolist())
    except Exception as exc:result.update(error_class=type(exc).__name__,reason=str(exc)[:300])
    result.update(finished_utc=now(),elapsed_seconds=round(time.monotonic()-started,3));dump(folder/'result.json',result)


def run_one(task):
    folder=ROOT/'docking/jobs'/task['job_id'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return read(folder/'result.json')
    if (folder/'task.json').exists():assert read(folder/'task.json')==task
    else:dump(folder/'task.json',task)
    started=time.monotonic()
    with (folder/'worker.log').open('a') as stream:
        try:
            result=subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--worker',task['job_id']],
                cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            if (folder/'result.json').exists():return read(folder/'result.json')
            output=dict(job_id=task['job_id'],chemblid=task['chemblid'],canonical=task['canonical'],status='worker_failed',vina=None,returncode=result.returncode)
        except subprocess.TimeoutExpired:
            output=dict(job_id=task['job_id'],chemblid=task['chemblid'],canonical=task['canonical'],status='timeout',vina=None)
    output.update(finished_utc=now(),elapsed_seconds=round(time.monotonic()-started,3));dump(folder/'result.json',output);return output


def write_observations(observations,jobs,pilot_only):
    for row in observations:
        job=row.get('job_id');path=ROOT/'docking/jobs'/str(job)/'result.json'
        if job and path.exists():
            result=read(path);row.update(docking_status=result['status'],vina=result.get('vina'),mmff_status=result.get('mmff_status'))
    name='pilot_observations.csv' if pilot_only else 'observations1175.csv'
    pd.DataFrame(observations).to_csv(ROOT/'docking'/name,index=False)
    complete=sum((ROOT/'docking/jobs'/job/'result.json').exists() for job in jobs)
    success=sum(read(ROOT/'docking/jobs'/job/'result.json').get('status')=='ok' for job in jobs if (ROOT/'docking/jobs'/job/'result.json').exists())
    progress=dict(time_utc=now(),pilot_only=pilot_only,unique_jobs=len(jobs),completed_unique_jobs=complete,successful_unique_jobs=success,
        generated_positions_observed=sum(row['generation_status']!='pending_generation' for row in observations if row['kind']=='generated'),
        observation_records=len(observations),new_API_calls=0)
    dump(ROOT/'logs/docking_progress.json',progress);return progress


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker');p.add_argument('--watch',action='store_true');p.add_argument('--pilot-once',action='store_true');args=p.parse_args()
    if args.worker:worker(args.worker);return
    (ROOT/'docking').mkdir(exist_ok=True)
    guard=(ROOT/'logs/docking.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not args.pilot_once:assert read(ROOT/'analysis/pilot_review.json')['allow_full_run'] is True
    dump(ROOT/'docking/protocol.json',protocol())
    active={};seen=set()
    with ThreadPoolExecutor(max_workers=2 if args.pilot_once else 8) as pool:
        while True:
            observations,jobs=collect(args.pilot_once)
            for job,task in jobs.items():
                if job not in seen and not (ROOT/'docking/jobs'/job/'result.json').exists():
                    active[pool.submit(run_one,task)]=job;seen.add(job)
            if active:
                done,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
                for f in done:
                    job=active.pop(f);result=f.result();print('DOCK',job,result['status'],result.get('vina'),flush=True)
            status=write_observations(observations,jobs,args.pilot_once)
            generated_finished=status['generated_positions_observed']==(16 if args.pilot_once else 1000)
            if not active and (generated_finished or not args.watch):break
            if not active:time.sleep(20)
    dump(ROOT/'docking/complete.json',dict(**status,status='complete' if generated_finished else 'snapshot_complete'))
    print('DOCKING STATUS',json.dumps(status),flush=True)


if __name__=='__main__':main()
