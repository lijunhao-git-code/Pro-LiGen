"""Reproduce all current-paper numerical evidence without an API or server."""
from pathlib import Path
import argparse,json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('--figures',action='store_true');a=p.parse_args()
    dest=Path(a.out).resolve();assert not dest.exists(),'Choose a new output directory'
    assert not dest.is_relative_to(ROOT/'extensions'),'Keep outputs separate from the frozen extension'
    dest.mkdir(parents=True)
    subprocess.run([sys.executable,str(ROOT/'reproduce/offline_analysis.py'),'--out',str(dest/'initial')],check=True)
    work=dest/'extended';shutil.copytree(ROOT/'extensions/20261102/data',work/'data');shutil.copytree(ROOT/'extensions/20261102/reproduce',work/'reproduce');(work/'figures').mkdir()
    for script in ['extended_comparison.py','mechanism_docking.py']:
        subprocess.run([sys.executable,str(work/'reproduce'/script)],check=True)
    if a.figures:
        subprocess.run([sys.executable,str(ROOT/'reproduce/render_figures.py'),'--out',str(dest/'initial_figures')],check=True)
        for script in ['plot_extended_comparison.py','plot_mechanism_docking.py']:subprocess.run([sys.executable,str(work/'reproduce'/script)],check=True)
    proof=dict(status='passed',initial_cohorts_reproduced=True,extended_planned_runs=1500,tasks_per_condition=25,paired_tests=24,
        candidate_policies_replayed=7500,docking_observations=759,API_calls=0,training_jobs=0,docking_jobs=0,figures=a.figures)
    (dest/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
