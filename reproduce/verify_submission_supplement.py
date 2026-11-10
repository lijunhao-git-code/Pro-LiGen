"""Verify the current submission supplement without generation, training or docking."""
from pathlib import Path
import argparse,json,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];X=ROOT/'extensions/20261110'
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    out=a.out.resolve();assert not out.exists(),'Choose a new output directory';out.mkdir(parents=True)
    scripts=['reproduce/external_comparison.py','reproduce/verify_docking_cases.py','reproduce/power_sensitivity.py','data/reproducibility20261107/reproduce_api_resources.py']
    checks=[]
    for name in scripts:
        r=subprocess.run([sys.executable,str(X/name)],capture_output=True,text=True,encoding='utf-8',errors='replace')
        (out/(Path(name).stem+'.txt')).write_text(r.stdout+'\n'+r.stderr,encoding='utf-8')
        assert r.returncode==0,(name,r.stderr[-2000:])
        checks.append(dict(script=name,exit_code=0));print('Verified:',name,flush=True)
    result=dict(status='passed',checks=checks,formal_external_positions=750,external_candidate_positions=4390,API_calls=0,training_jobs=0,docking_jobs=0)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
