"""Validate the retained GIN training data, log, curve and deployed source; no training."""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        parser.error('--out must be new')
    from rdkit import Chem, rdBase
    paths = {name: ROOT / name for name in [
        'data/training/triplets_ring_count.csv', 'data/training/best_gin_rc.csv',
        'data/training/train_rc.log.txt', 'archive/original_project/run.py',
        'proligen/gin.py', 'archive/train_gin.py', 'models/best_gin_rc.pth',
        'data/training/source_smiles26402.csv','data/retrieval/library.csv']}
    before = {name: sha(path) for name, path in paths.items()}
    triplets = rows(paths['data/training/triplets_ring_count.csv'])
    curve = rows(paths['data/training/best_gin_rc.csv'])
    assert len(triplets) == 15461
    assert len(curve) == 50 and [int(row['epoch']) for row in curve] == list(range(1, 51))
    source=rows(paths['data/training/source_smiles26402.csv'])
    assert len(source)==26402
    originals={row['smiles'] for row in source}
    canonical=set()
    for smiles in originals:
        mol=Chem.MolFromSmiles(smiles)
        assert mol is not None
        canonical.add(Chem.MolToSmiles(mol,isomericSmiles=True))
    assert len(canonical)==26108
    assert canonical=={row['canonical'] for row in rows(paths['data/retrieval/library.csv'])}
    cache = {}
    for i, row in enumerate(triplets):
        for role in ('anchor', 'positive', 'negative'):
            smiles = row[role + '_smiles']
            assert smiles in originals
            if smiles not in cache:
                mol = Chem.MolFromSmiles(smiles)
                assert mol is not None, (i, role)
                cache[smiles] = mol.GetRingInfo().NumRings()
            assert cache[smiles] == int(float(row[role + '_ring_count'])), (i, role)
        assert float(row['positive_ring_count']) <= float(row['negative_ring_count'])
    log = paths['data/training/train_rc.log.txt'].read_text(encoding='utf-8', errors='replace')
    assert 'Total: 15461 | Train: 12368 | Val: 3093' in log
    parsed = re.findall(r'Epoch\s+(\d+)\s*\|\s*Train:\s*([0-9.eE+-]+)\s*\|\s*Val:\s*([0-9.eE+-]+)', log)
    assert len(parsed) == 50
    for row, (epoch, train_loss, val_loss) in zip(curve, parsed):
        assert int(row['epoch']) == int(epoch)
        assert abs(float(row['train_loss']) - float(train_loss)) <= 0.000051
        assert abs(float(row['val_loss']) - float(val_loss)) <= 0.000051
    def definitions(path):
        return {node.name: node for node in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    original = definitions(paths['archive/original_project/run.py'])
    restored = definitions(paths['proligen/gin.py'])
    names = ['atom_features', 'bond_features', 'mol_to_graph', 'GINEncoder']
    assert all(ast.dump(original[name]) == ast.dump(restored[name]) for name in names)
    assert before['models/best_gin_rc.pth'] == 'b08cf0fa0c2625451711d42336f1c032adc6974933d905e4f526471183182253'
    assert all(sha(path) == before[name] for name, path in paths.items())
    report = dict(status='passed', triplets_checked=len(triplets), unique_smiles_checked=len(cache),
        source_smiles_rows=len(source),source_canonical_unique=len(canonical),
        source_canonical_set_matches_released_library=True,triplet_structures_all_present_in_source=True,
        ring_count_labels_all_match=True, positive_negative_property_order_all_match=True,
        retained_curve_epochs=len(curve), retained_log_epochs=len(parsed),
        log_curve_agree_at_printed_precision=True, logged_train_triplets=12368,
        logged_validation_triplets=3093, all_four_functions_match_original_deployed_source=True,
        rdkit=rdBase.rdkitVersion, inputs_unchanged=True, input_sha256=before,
        new_training_jobs=0, language_model_API_calls=0, docking_jobs=0,
        scope='Validate retained training artifacts and code; does not reconstruct original random split indices')
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['status','triplets_checked','unique_smiles_checked','retained_curve_epochs',
        'all_four_functions_match_original_deployed_source']}, indent=2))


if __name__ == '__main__':
    main()
