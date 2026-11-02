"""Recompute this offline audit in a new directory without overwriting delivered results."""
import argparse,hashlib,json,shutil,subprocess,sys
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,help='New output directory that must not exist')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];out=Path(args.output).resolve()
    if out.exists():parser.error('--output must be a new directory')
    frozen=json.loads((root/'frozen_inputs_and_code.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((root/name).read_bytes()).hexdigest()==v for name,v in frozen.items())
    out.mkdir(parents=True)
    for name in list(frozen)+['frozen_inputs_and_code.json']:
        target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/name,target)
    for name in ['figures.py','audit.py','report.py','reproduce.py']:
        shutil.copy2(root/'code'/name,out/'code'/name)
    for name in ['analyze.py','figures.py','audit.py','report.py']:
        subprocess.run([sys.executable,str(out/'code'/name)],check=True)
    print('Completed fresh offline rerun:',out)


if __name__=='__main__':main()
