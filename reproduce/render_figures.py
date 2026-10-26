"""Render quantitative figures into a new directory from the released frozen inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
from figure_text_overrides import apply_editorial_labels

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    destination = args.out.resolve()
    assert not destination.exists(), 'Choose a new output directory'
    spec = importlib.util.spec_from_file_location('retained_figure_implementation', ROOT / 'audit/b05_make_figures.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    receipt = apply_editorial_labels(renderer)
    inputs = json.loads((ROOT / 'reproduce/figure_inputs_frozen.json').read_text(encoding='utf-8'))
    destination.mkdir(parents=True, exist_ok=False)
    for name in ('history', 'mechanism', 'downstream', 'docking', 'reference', 'chemistry'):
        getattr(renderer, name)(inputs, destination)
    (destination / 'text_override_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print('Six quantitative figure assets rendered from frozen scientific inputs.')

if __name__ == '__main__':
    main()
