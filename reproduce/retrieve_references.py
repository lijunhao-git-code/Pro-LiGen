"""Reproduce Morgan and frozen-GIN top-three references without model APIs.

Default: recompute seed embeddings and use the released library embeddings.
--recompute-library: also build and encode all library graphs from SMILES.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CHECKPOINT_SHA256 = 'b08cf0fa0c2625451711d42336f1c032adc6974933d905e4f526471183182253'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def encode(model, smiles, batch_size):
    import torch
    from torch_geometric.data import Batch
    from proligen.gin import mol_to_graph
    result = []
    with torch.inference_mode():
        for offset in range(0, len(smiles), batch_size):
            graphs = [mol_to_graph(value) for value in smiles[offset:offset + batch_size]]
            if any(graph is None or graph.x.shape[0] == 0 for graph in graphs):
                raise ValueError('Invalid or empty molecule in the fixed inference inputs')
            data = Batch.from_data_list(graphs)
            result.append(torch.nn.functional.normalize(model(data), dim=1).cpu().numpy())
            if len(smiles) > 128 and (offset == 0 or offset % (batch_size * 20) == 0):
                print(f'Encoded {min(offset + batch_size, len(smiles))}/{len(smiles)} library molecules', flush=True)
    return np.concatenate(result)


def top_three(scores, eligible, canonical):
    indices = np.flatnonzero(eligible & np.isfinite(scores))
    if len(indices) < 3:
        raise ValueError('Fewer than three candidates in the fixed exact-ring pool')
    # Same ordering as the original partition-then-lexsort selector, including ties.
    return indices[np.lexsort((canonical[indices], -scores[indices]))[:3]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path, help='A NEW directory for results')
    parser.add_argument('--task', help='Optional single ChEMBL task; default is all 25 tasks')
    parser.add_argument('--recompute-library', action='store_true', help='Encode all 26,108 library SMILES again')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    destination = args.out.resolve()
    if destination.exists():
        parser.error('--out must name a new directory; existing outputs are never overwritten')
    if args.batch_size < 1 or args.threads < 1:
        parser.error('--batch-size and --threads must be positive')

    import torch
    import torch_geometric
    from rdkit import Chem, DataStructs, rdBase
    from rdkit.Chem import rdFingerprintGenerator
    from proligen.gin import GINEncoder

    started = time.monotonic()
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    paths = {
        'checkpoint': ROOT / 'models/best_gin_rc.pth',
        'library': ROOT / 'data/retrieval/library.csv',
        'library_embeddings': ROOT / 'data/retrieval/library_embeddings.npy',
        'tasks': ROOT / 'data/SI_records/tasks25.json',
        'references': ROOT / 'data/SI_records/B_references150.json',
        'encoder': ROOT / 'proligen/gin.py',
    }
    before = {name: sha(path) for name, path in paths.items()}
    if before['checkpoint'] != CHECKPOINT_SHA256:
        raise ValueError('Checkpoint hash differs from the deployed Study-B checkpoint')
    with paths['library'].open(encoding='utf-8', newline='') as stream:
        library = list(csv.DictReader(stream))
    canonical = np.asarray([row['canonical'] for row in library], dtype=str)
    rings = np.asarray([int(row['RingCount']) for row in library])
    if len(library) != 26108 or len(set(canonical)) != 26108:
        raise ValueError('The released library must contain 26,108 unique canonical structures')
    tasks = read_json(paths['tasks'])
    if len(tasks) != 25 or len({row['chemblid'] for row in tasks}) != 25:
        raise ValueError('The released benchmark requires 25 unique tasks')
    if args.task:
        tasks = [row for row in tasks if row['chemblid'] == args.task]
        if not tasks:
            parser.error('Unknown task identifier')
    expected = {(row['chemblid'], row['group'], row['rank']): row
                for row in read_json(paths['references'])}

    model = GINEncoder(in_dim=84, hidden_dim=256, out_dim=256, num_layers=5)
    model.load_state_dict(torch.load(paths['checkpoint'], map_location='cpu', weights_only=True), strict=True)
    model.eval()
    cached = np.load(paths['library_embeddings'], mmap_mode='r')
    if cached.shape != (26108, 256) or not np.isfinite(cached).all():
        raise ValueError('Invalid released library embeddings')
    if args.recompute_library:
        print('Recomputing library graph embeddings on CPU; no training.', flush=True)
        embeddings = encode(model, canonical.tolist(), args.batch_size)
        embedding_difference = float(np.max(np.abs(embeddings - cached)))
    else:
        embeddings = cached
        embedding_difference = None
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1., atol=1e-6, rtol=0):
        raise ValueError('Library embeddings are not L2-normalized')

    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
    fingerprints = []
    for value in canonical:
        mol = Chem.MolFromSmiles(str(value))
        if mol is None:
            raise ValueError('Unparseable molecule in the released library')
        fingerprints.append(fpgen.GetFingerprint(mol))
    counts = np.asarray([fp.GetNumOnBits() for fp in fingerprints], dtype=float)
    selected, task_checks, mismatches = [], [], []
    maxima = {'M': 0., 'G': 0.}
    for task in tasks:
        mol = Chem.MolFromSmiles(task['seed_smiles'])
        if mol is None:
            raise ValueError('Unparseable seed in released tasks')
        qfp = fpgen.GetFingerprint(mol)
        tanimoto = np.asarray(DataStructs.BulkTanimotoSimilarity(qfp, fingerprints))
        shared_bits = np.rint(tanimoto * (qfp.GetNumOnBits() + counts) / (1. + tanimoto))
        m_scores = shared_bits / np.sqrt(np.maximum(qfp.GetNumOnBits() * counts, 1e-10))
        query = encode(model, [task['seed_smiles']], 1)[0]
        g_scores = embeddings @ query
        eligible = (rings == task['target_rc']) & (canonical != task['seed_canonical'])
        matched = {'M': True, 'G': True}
        for group, scores in [('M', m_scores), ('G', g_scores)]:
            for rank, index in enumerate(top_three(scores, eligible, canonical), 1):
                row = dict(chemblid=task['chemblid'], group=group, rank=rank,
                           library_id=library[index]['query_id'], canonical=str(canonical[index]),
                           RC=int(rings[index]), similarity=float(scores[index]))
                reference = expected[(row['chemblid'], group, rank)]
                agrees = row['canonical'] == reference['canonical'] and row['library_id'] == reference['library_id']
                if not agrees:
                    matched[group] = False
                    mismatches.append(dict(task=row['chemblid'], group=group, rank=rank,
                                           actual=row['canonical'], expected=reference['canonical']))
                maxima[group] = max(maxima[group], abs(row['similarity'] - reference['similarity']))
                selected.append(row)
        task_checks.append(dict(chemblid=task['chemblid'], pool_size=int(eligible.sum()),
                                Morgan_top3_exact=matched['M'], GIN_top3_exact=matched['G']))
        print(f"Checked {task['chemblid']}: Morgan={matched['M']}, GIN={matched['G']}", flush=True)
    if any(sha(path) != before[name] for name, path in paths.items()):
        raise RuntimeError('An inference input changed during the run')
    passed = not mismatches and max(maxima.values()) <= 1e-6
    if embedding_difference is not None:
        passed = passed and embedding_difference <= 1e-6
    report = dict(status='passed' if passed else 'failed', tasks_checked=len(tasks), references_checked=len(selected),
        all_Morgan_top3_exact=all(row['Morgan_top3_exact'] for row in task_checks),
        all_GIN_top3_exact=all(row['GIN_top3_exact'] for row in task_checks),
        library_recomputed=args.recompute_library, library_size=len(library), embedding_dimension=256,
        library_embedding_max_abs_difference=embedding_difference,
        saved_reference_score_max_abs_difference=maxima, score_absolute_tolerance=1e-6,
        per_task=task_checks, mismatches=mismatches, inputs_unchanged=True,
        input_sha256={paths[name].relative_to(ROOT).as_posix(): digest for name, digest in before.items()},
        versions=dict(python=platform.python_version(), torch=torch.__version__,
                      torch_geometric=torch_geometric.__version__, rdkit=rdBase.rdkitVersion, numpy=np.__version__),
        device='cpu', batch_size=args.batch_size, threads=args.threads,
        elapsed_seconds=round(time.monotonic() - started, 3),
        training_jobs=0, language_model_API_calls=0, docking_jobs=0)
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'references.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    (destination / 'verification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (destination / 'references.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0]))
        writer.writeheader(); writer.writerows(selected)
    print(json.dumps({key: report[key] for key in ['status', 'tasks_checked', 'references_checked',
          'library_recomputed', 'library_embedding_max_abs_difference', 'saved_reference_score_max_abs_difference']}, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
