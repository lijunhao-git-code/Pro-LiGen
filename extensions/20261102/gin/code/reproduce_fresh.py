"""Recompute numeric results and plotted coordinates in a new independent directory."""
import hashlib,json,os,shutil,sqlite3,subprocess,sys,time
from pathlib import Path
from common import ROOT,read,sha,dump,now


def main():
    assert read(ROOT/'logs/offline_pipeline_status.json')['status']=='analyses_and_figures_complete'
    stamp=now().replace(':','').replace('+','_');fresh=ROOT/'reproduce'/('fresh_'+stamp);fresh.mkdir(parents=True)
    files=set()
    for name in read(ROOT/'inputs/generation_locked.json')['files']:files.add(ROOT/name)
    for folder in ['code','source','requests','responses','results']:
        files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for prop in ['logp','tpsa']:
        for split in ['train','val']:files.add(ROOT/f'training/{prop}_{split}_triplets.csv')
    files.add(ROOT/'inputs/generation_locked.json')
    files.update((ROOT/'analysis/mcs_cache').glob('*.json'))
    files.add(ROOT/'docking/observations1175.csv')
    files.add(ROOT/'docking/complete.json')
    for folder in (ROOT/'docking/jobs').iterdir():
        if folder.is_dir():
            for name in ['task.json','result.json','prepared_ligand.sdf','ligand.pdbqt','poses.pdbqt']:
                if (folder/name).exists():files.add(folder/name)
    before={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    for p in sorted(files):
        target=fresh/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    (fresh/'logs').mkdir(exist_ok=True)
    source=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True);dest=sqlite3.connect(fresh/'logs/usage.sqlite');source.backup(dest);dest.close();source.close()
    for args in [['code/analyze_properties.py'],['code/analyze_structure_docking.py'],['code/audit_docking_sources.py'],['code/plot_results.py']]:
        p=subprocess.run([sys.executable,'-u',*args],cwd=fresh)
        if p.returncode:raise RuntimeError('Fresh analysis failed: '+args[0])
    compared=[]
    for directory in ['properties','structures','docking','figures']:
        for p in sorted((ROOT/'analysis'/directory).glob('*.csv')):
            candidate=fresh/p.relative_to(ROOT);assert candidate.is_file()
            equal=sha(p)==sha(candidate);assert equal,'Fresh numeric mismatch: '+str(p.relative_to(ROOT))
            compared.append(dict(file=p.relative_to(ROOT).as_posix(),sha256=sha(p),fresh_sha256=sha(candidate),identical=True))
    assert all(sha(ROOT/name)==value for name,value in before.items())
    result=dict(status='passed',time_utc=now(),fresh_directory=str(fresh.relative_to(ROOT)),numeric_files_exact=len(compared),
        plotted_coordinates_exact=True,all_source_files_unchanged=True,new_API_calls=0,new_training_jobs=0,new_docking_jobs=0,
        comparisons=compared)
    dump(ROOT/'analysis/fresh_reproduction_verification.json',result);print('FRESH REPRODUCTION',len(compared),'files identical',flush=True)


if __name__=='__main__':main()
