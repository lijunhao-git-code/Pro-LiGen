"""After training, validate and run only the16-position pilot. Never launches full generation."""
import subprocess,sys,time,traceback
from common import ROOT,read,dump,now


def main():
    try:
        dump(ROOT/'logs/pilot_pipeline_status.json',dict(stage='waiting_for_property_models',status='running',time_utc=now()))
        while not (ROOT/'training/complete.json').exists():
            state=read(ROOT/'logs/training_pipeline_status.json')
            if state.get('status')=='failed':raise RuntimeError('Training failed; pilot not started')
            time.sleep(15)
        for stage,script,args in [('self_tests','test_runner.py',[]),('free_API_probe','service.py',[]),
                                  ('freeze_generation_inputs','prepare_generation.py',[]),('pilot16','run_generation.py',['--pilot'])]:
            dump(ROOT/'logs/pilot_pipeline_status.json',dict(stage=stage,status='running',time_utc=now()))
            subprocess.run([sys.executable,'-u',str(ROOT/'code'/script),*args],cwd=ROOT,check=True)
        dump(ROOT/'logs/pilot_pipeline_status.json',dict(stage='pilot_completed_awaiting_review',status='complete',time_utc=now(),full_generation_started=False))
    except Exception as exc:
        dump(ROOT/'logs/pilot_pipeline_status.json',dict(stage='pilot_pipeline_stopped',status='stopped',error_class=type(exc).__name__,
            time_utc=now(),full_generation_started=False))
        traceback.print_exc();raise


if __name__=='__main__':main()
