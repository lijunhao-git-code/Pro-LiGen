"""Rerun the frozen numerical analysis in a new output directory, leaving supplied results intact."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory that must not already exist')
    parser.add_argument('--figures', action='store_true', help='Also render supplementary PNG/SVG figures')
    parser.add_argument('--audit', action='store_true', help='Also run independent checks (requires --figures)')
    parser.add_argument('--gallery', action='store_true', help='Also draw the offline molecular inspection gallery')
    args = parser.parse_args()
    if args.audit and not args.figures:
        parser.error('--audit requires --figures')
    root = Path(__file__).resolve().parents[1]
    out = Path(args.output).resolve()
    if out.exists():
        parser.error('--output must be a new directory; supplied data/results will not be overwritten')
    frozen = json.loads((root/'frozen_inputs_and_code.json').read_text(encoding='utf-8'))
    for rel, expected in frozen.items():
        if hashlib.sha256((root/rel).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Input/code checksum mismatch: '+rel)
    out.mkdir(parents=True)
    for rel in list(frozen) + ['frozen_inputs_and_code.json', 'requirements.txt']:
        target = out/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/rel, target)
    for name in ['report_figures.py', 'reproduce_fresh.py', 'gallery.py', 'audit.py', 'write_report.py']:
        if (root/'code'/name).exists():
            shutil.copy2(root/'code'/name, out/'code'/name)
    subprocess.run([sys.executable, str(out/'code/analyze.py')], check=True)
    if args.figures:
        subprocess.run([sys.executable, str(out/'code/report_figures.py')], check=True)
    if args.audit:
        subprocess.run([sys.executable, str(out/'code/audit.py')], check=True)
        subprocess.run([sys.executable, str(out/'code/write_report.py')], check=True)
    if args.gallery:
        subprocess.run([sys.executable, str(out/'code/gallery.py')], check=True)
    print('New independent run:', out)


if __name__ == '__main__':
    main()
