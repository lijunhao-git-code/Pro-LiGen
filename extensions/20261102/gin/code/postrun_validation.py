"""Run independent count and fresh-directory verification once all offline stages finish."""
import subprocess,sys,time
from common import ROOT,read,dump,now


def main():
    while True:
        p=ROOT/'logs/offline_pipeline_status.json'
        if p.exists() and read(p).get('status')=='analyses_and_figures_complete':break
        g=read(ROOT/'logs/generation_status.json')
        if g.get('stop_reason') and g['completed']<1000:
            dump(ROOT/'logs/postrun_validation_status.json',dict(status='waiting_for_generation_resume',reason=g['stop_reason'],time_utc=now()));return
        time.sleep(20)
    for args in [['code/audit_final_counts.py'],['code/audit_docking_sources.py'],['code/reproduce_fresh.py']]:
        p=subprocess.run([sys.executable,'-u',*args],cwd=ROOT)
        if p.returncode:
            dump(ROOT/'logs/postrun_validation_status.json',dict(status='failed',stage=args[0],returncode=p.returncode,time_utc=now()));return
    dump(ROOT/'logs/postrun_validation_status.json',dict(status='passed_ready_for_report',time_utc=now()))
    print('POSTRUN VALIDATION COMPLETE',flush=True)


if __name__=='__main__':main()
