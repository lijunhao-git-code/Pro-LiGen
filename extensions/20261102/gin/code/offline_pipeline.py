"""Wait for the authorized run, then produce auditable offline analyses and figures."""
import os,subprocess,sys,time
from common import ROOT,read,dump,now


def execute(args):
    result=subprocess.run([sys.executable,'-u',*args],cwd=ROOT)
    if result.returncode:raise RuntimeError('Offline stage failed: '+args[0])


def main():
    while True:
        status=read(ROOT/'logs/generation_status.json')
        if status['completed']==1000:break
        if status.get('stop_reason'):
            dump(ROOT/'logs/offline_pipeline_status.json',dict(status='generation_paused',reason=status['stop_reason'],time_utc=now(),completed=status['completed']))
            execute(['code/analyze_properties.py','--partial','--output','analysis/paused_snapshot']);return
        dump(ROOT/'logs/offline_pipeline_status.json',dict(status='waiting_for_generation',time_utc=now(),completed=status['completed']))
        time.sleep(20)
    execute(['code/analyze_properties.py'])
    while True:
        p=ROOT/'analysis/mcs_complete.json'
        if p.exists() and read(p)['completed_positions']==1000:break
        dump(ROOT/'logs/offline_pipeline_status.json',dict(status='waiting_for_MCS',time_utc=now()));time.sleep(20)
    execute(['code/analyze_structure_docking.py','--structure-only'])
    execute(['code/plot_results.py'])
    while True:
        p=ROOT/'docking/complete.json'
        if p.exists() and read(p).get('status')=='complete':break
        dump(ROOT/'logs/offline_pipeline_status.json',dict(status='waiting_for_docking',time_utc=now()));time.sleep(20)
    execute(['code/analyze_structure_docking.py','--docking-only']);execute(['code/plot_results.py'])
    dump(ROOT/'logs/offline_pipeline_status.json',dict(status='analyses_and_figures_complete',time_utc=now()))
    print('OFFLINE PIPELINE COMPLETE',flush=True)


if __name__=='__main__':main()
