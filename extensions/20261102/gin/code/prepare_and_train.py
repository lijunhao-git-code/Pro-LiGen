"""Durable sequential data preparation and property-model training."""
import subprocess,sys,traceback
from common import ROOT,dump,now


def main():
    try:
        for stage,script in [('prepare_data','prepare_data.py'),('train_models','train_models.py')]:
            dump(ROOT/'logs/training_pipeline_status.json',dict(stage=stage,status='running',time_utc=now(),new_API_calls=0))
            subprocess.run([sys.executable,'-u',str(ROOT/'code'/script)],cwd=ROOT,check=True)
        dump(ROOT/'logs/training_pipeline_status.json',dict(stage='training_complete',status='complete',time_utc=now(),new_API_calls=0))
    except Exception as exc:
        dump(ROOT/'logs/training_pipeline_status.json',dict(stage='preparation_or_training_failed',status='failed',error_class=type(exc).__name__,time_utc=now(),new_API_calls=0))
        traceback.print_exc();raise


if __name__=='__main__':main()
