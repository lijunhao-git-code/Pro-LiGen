"""Complete offline validation and packaging after the authorized generation job exits."""
from pathlib import Path
import json,os,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[1]
def write(x):(ROOT/'logs/completion_status.json').write_text(json.dumps(x,indent=2)+'\n')
def main():
 pid=json.loads((ROOT/'logs/full_process.json').read_text())['pid']
 while Path('/proc/'+str(pid)).exists():time.sleep(15)
 status=json.loads((ROOT/'logs/generation_status.json').read_text())
 if status['completed']!=750:
  write(dict(status='generation_paused_or_incomplete',completed=status['completed'],reason=status.get('stop_reason')));return
 for script in ['audit.py','analyze.py','write_report.py','package.py']:
  write(dict(status='running_offline_stage',script=script))
  p=subprocess.run([sys.executable,str(ROOT/'code'/script)],cwd=ROOT,capture_output=True,text=True)
  (ROOT/'logs'/(script+'.log')).write_text(p.stdout+'\n'+p.stderr)
  if p.returncode:write(dict(status='offline_stage_failed',script=script,error_class='nonzero_exit',returncode=p.returncode));return
 write(dict(status='complete',formal_positions=750,new_training_jobs=0,new_docking_jobs=0))
if __name__=='__main__':main()
