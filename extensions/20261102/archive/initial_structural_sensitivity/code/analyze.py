"""Standalone offline retrieval baselines and structural audit; no network/API calls."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import Descriptors, QED, rdFingerprintGenerator, rdFMCS
from rdkit.Contrib.SA_Score import sascorer

from retained_statistics import paired, correct_holm

ROOT = Path(__file__).resolve().parents[1]
METHODS = ['T', 'M', 'G', 'R_NN1', 'R_QED3', 'R_QED6']
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=False)
RDLogger.DisableLog('rdApp.warning')
RDLogger.DisableLog('rdApp.error')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read(name):
    return json.loads((ROOT / 'inputs' / name).read_text(encoding='utf-8'))


def save_csv(name, records):
    pd.DataFrame(records).to_csv(ROOT / 'results' / name, index=False, encoding='utf-8-sig')


def canonical(mol, stereo=True):
    if not stereo:
        mol = Chem.Mol(mol)
        Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def organic_component(mol):
    parts = Chem.GetMolFrags(mol, asMols=True)
    organic = [p for p in parts if any(a.GetAtomicNum() == 6 for a in p.GetAtoms())]
    if len(organic) != 1:
        return None
    other = [p for p in parts if not any(a.GetAtomicNum() == 6 for a in p.GetAtoms())]
    if any(p.GetNumAtoms() != 1 for p in other):
        return None
    return organic[0]


@lru_cache(maxsize=None)
def describe(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        raise ValueError('Invalid molecular input')
    core = organic_component(mol)
    return dict(mol=mol, canonical=canonical(mol), nostereo=canonical(mol, False),
                RC=mol.GetRingInfo().NumRings(), QED=float(QED.qed(mol)),
                SA_raw=float(sascorer.calculateScore(mol)), MW=float(Descriptors.MolWt(mol)),
                HeavyAtoms=mol.GetNumHeavyAtoms(), fragments=len(Chem.GetMolFrags(mol)),
                fp=FPGEN.GetFingerprint(mol), organic=canonical(core) if core is not None else None,
                organic_mol=core, organic_fp=FPGEN.GetFingerprint(core) if core is not None else None,
                organic_QED=float(QED.qed(core)) if core is not None else None,
                organic_MW=float(Descriptors.MolWt(core)) if core is not None else None,
                organic_HA=core.GetNumHeavyAtoms() if core is not None else None,
                organic_logP=float(Descriptors.MolLogP(core)) if core is not None else None,
                organic_TPSA=float(Descriptors.TPSA(core)) if core is not None else None,
                organic_HBD=Descriptors.NumHDonors(core) if core is not None else None,
                organic_HBA=Descriptors.NumHAcceptors(core) if core is not None else None)


def mcs_key(seed, candidate):
    return hashlib.sha256((seed + '\0' + candidate).encode()).hexdigest()


def mcs_worker(job):
    key, seed, candidate, timeouts = job
    a, b = Chem.MolFromSmiles(seed), Chem.MolFromSmiles(candidate)
    started = time.monotonic()
    result = None
    for timeout in timeouts:
        result = rdFMCS.FindMCS([a, b], maximizeBonds=False, timeout=timeout,
            matchValences=False, ringMatchesRingOnly=False, completeRingsOnly=False,
            matchChiralTag=False, atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder)
        if not result.canceled:
            break
    assert result.numAtoms <= min(a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms())
    return dict(key=key, seed_organic=seed, candidate_organic=candidate,
                common_atoms=result.numAtoms, common_bonds=result.numBonds,
                complete=not result.canceled, smarts=result.smartsString,
                seed_fraction=result.numAtoms / a.GetNumHeavyAtoms(),
                candidate_fraction=result.numAtoms / b.GetNumHeavyAtoms(),
                upper_seed_fraction=min(a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms()) / a.GetNumHeavyAtoms(),
                elapsed_seconds=round(time.monotonic() - started, 4), last_timeout_seconds=timeout)


def compute_mcs(pairs, protocol):
    path = ROOT / 'results/mcs_cache.jsonl'
    cache = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            cache[row['key']] = row
    jobs = [(key, seed, candidate, protocol['structures']['MCS_timeouts_seconds'])
            for key, (seed, candidate) in sorted(pairs.items()) if key not in cache]
    print(f'MCS pairs: {len(pairs)} unique, {len(jobs)} to compute.', flush=True)
    with path.open('a', encoding='utf-8') as stream:
        with ProcessPoolExecutor(max_workers=protocol['structures']['MCS_workers']) as pool:
            futures = [pool.submit(mcs_worker, job) for job in jobs]
            for i, future in enumerate(as_completed(futures), 1):
                row = future.result()
                cache[row['key']] = row
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                if i % 50 == 0 or i == len(jobs):
                    print(f'MCS {i}/{len(jobs)} computed; incomplete={sum(not r["complete"] for r in cache.values())}', flush=True)
    assert all(key in cache for key in pairs)
    return cache


def choose(items):
    eligible = {r['canonical']: r for r in sorted(items, key=lambda r: (r['turn'], r['candidate_position'])) if r['eligible']}
    if not eligible:
        return None
    return min(eligible.values(), key=lambda r: (-r['QED'], r['SA_raw'], r['canonical']))


def retrieve(tasks, library, reference):
    mols = [Chem.MolFromSmiles(s) for s in library.canonical]
    assert all(m is not None for m in mols)
    assert all(canonical(m) == s for m, s in zip(mols, library.canonical))
    assert all(m.GetRingInfo().NumRings() == rc for m, rc in zip(mols, library.RingCount))
    keys = library.canonical.to_numpy(str)
    rings = library.RingCount.to_numpy(int)
    singles = np.asarray([len(Chem.GetMolFrags(m)) == 1 for m in mols])
    fps = [FPGEN.GetFingerprint(m) for m in mols]
    counts = np.asarray([fp.GetNumOnBits() for fp in fps], float)
    reference_map = {(r['chemblid'], r['rank']): r for r in reference if r['group'] == 'M'}
    outputs, candidates, pools = [], [], []
    match_n = 0
    for task in tasks:
        seed = describe(task['seed_smiles'])
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(seed['fp'], fps))
        shared = np.rint(similarities * (seed['fp'].GetNumOnBits() + counts) / (1 + similarities))
        scores = shared / np.sqrt(np.maximum(seed['fp'].GetNumOnBits() * counts, 1e-10))
        original_pool = (rings == task['target_rc']) & (keys != task['seed_canonical'])
        idx = np.flatnonzero(original_pool)
        original_top3 = idx[np.lexsort((keys[idx], -scores[idx]))[:3]]
        for rank, index in enumerate(original_top3, 1):
            expected = reference_map[(task['chemblid'], rank)]
            assert keys[index] == expected['canonical']
            assert abs(scores[index] - expected['similarity']) < 1e-10
            match_n += 1
        eligible = original_pool & singles
        idx = np.flatnonzero(eligible)
        assert len(idx) >= 6
        top6 = idx[np.lexsort((keys[idx], -scores[idx]))[:6]]
        pools.append(dict(chemblid=task['chemblid'], target_rc=task['target_rc'],
                          original_exact_ring_pool=int(original_pool.sum()),
                          single_component_output_pool=int(eligible.sum()),
                          multicomponent_removed=int((original_pool & ~singles).sum())))
        options = []
        for rank, index in enumerate(top6, 1):
            props = describe(str(keys[index]))
            row = dict(chemblid=task['chemblid'], canonical=props['canonical'], smiles=props['canonical'],
                       library_id=str(library.iloc[index].query_id), retrieval_rank=rank,
                       cosine=float(scores[index]), turn=0, candidate_position=rank, eligible=True,
                       QED=props['QED'], SA_raw=props['SA_raw'], RC=props['RC'])
            assert props['fragments'] == 1 and props['RC'] == task['target_rc']
            options.append(row)
        for method, k in [('R_NN1', 1), ('R_QED3', 3), ('R_QED6', 6)]:
            subset = [dict(r, method=method, slot_id=task['chemblid'] + '__' + method, position=1) for r in options[:k]]
            candidates.extend(subset)
            outputs.append(dict(choose(subset)))
        assert outputs[-1]['QED'] + 1e-12 >= outputs[-2]['QED'] >= outputs[-3]['QED'] - 1e-12
    assert match_n == 75
    return outputs, candidates, pools, match_n


def features(task, smiles, cache, library_set, library_ns, training_set):
    seed = describe(task['seed_smiles'])
    out = describe(smiles)
    row = dict(canonical=out['canonical'], QED=out['QED'], SA_raw=out['SA_raw'],
        RC=out['RC'], fragments=out['fragments'], MW=out['MW'], HeavyAtoms=out['HeavyAtoms'],
        tanimoto_parent=float(DataStructs.TanimotoSimilarity(seed['fp'], out['fp'])),
        dQED_parent=out['QED'] - seed['QED'], dMW_parent=out['MW'] - seed['MW'],
        dHA_parent=out['HeavyAtoms'] - seed['HeavyAtoms'],
        novel_library_exact=out['canonical'] not in library_set,
        novel_library_ignoring_stereo=out['nostereo'] not in library_ns,
        novel_training_exact=out['canonical'] not in training_set,
        organic_canonical=out['organic'], seed_organic_canonical=seed['organic'])
    if out['organic'] is not None and seed['organic'] is not None:
        mcs = cache[mcs_key(seed['organic'], out['organic'])]
        row.update(tanimoto_organic=float(DataStructs.TanimotoSimilarity(seed['organic_fp'], out['organic_fp'])),
            mcs_seed_fraction=mcs['seed_fraction'] if mcs['complete'] else None,
            mcs_output_fraction=mcs['candidate_fraction'] if mcs['complete'] else None,
            mcs_lower_seed_fraction=mcs['seed_fraction'], mcs_upper_seed_fraction=mcs['upper_seed_fraction'],
            mcs_complete=mcs['complete'], mcs_key=mcs['key'],
            dQED_organic=out['organic_QED'] - seed['organic_QED'],
            dMW_organic=out['organic_MW'] - seed['organic_MW'],
            dHA_organic=out['organic_HA'] - seed['organic_HA'],
            dlogP_organic=out['organic_logP'] - seed['organic_logP'],
            dTPSA_organic=out['organic_TPSA'] - seed['organic_TPSA'],
            dHBD_organic=out['organic_HBD'] - seed['organic_HBD'],
            dHBA_organic=out['organic_HBA'] - seed['organic_HBA'])
    return row


METRICS = ['delivery', 'yield_QED', 'QED', 'SA_raw', 'tanimoto_parent', 'tanimoto_organic',
           'mcs_seed_fraction', 'mcs_output_fraction', 'dQED_parent', 'dQED_organic',
           'dMW_parent', 'dHA_parent', 'dMW_organic', 'dHA_organic', 'dlogP_organic',
           'dTPSA_organic', 'dHBD_organic', 'dHBA_organic', 'novel_library_exact',
           'novel_library_ignoring_stereo', 'novel_training_exact']


def aggregate(rows):
    frame = pd.DataFrame(rows)
    for name in METRICS:
        if name not in frame:
            frame[name] = np.nan
        frame[name] = pd.to_numeric(frame[name], errors='coerce')
    targets = frame.groupby(['method', 'chemblid'], sort=True)[METRICS].mean().reset_index()
    summary = []
    for method in METHODS:
        part = frame[frame.method == method]
        means = targets[targets.method == method]
        assert len(means) == 25 and len(part) == (125 if method in ['T', 'M', 'G'] else 25)
        values = means[METRICS].mean().to_dict()
        summary.append(dict(method=method, planned=len(part), delivered=int(part.delivery.sum()), targets=25,
                            unique_outputs=part.canonical.dropna().nunique(), **values))
    return targets.to_dict('records'), summary


def gated_analysis(records, candidate_rows, feature_map, tasks, protocol):
    all_candidates = defaultdict(list)
    for row in candidate_rows:
        all_candidates[row['slot_id']].append(row)
    records_by_slot = {r['slot_id']: r for r in records}
    assert len(records_by_slot) == 450
    target_rows, summaries, traces = [], [], []
    for metric in ['tanimoto_parent', 'mcs_seed_fraction']:
        for threshold in protocol['structures']['thresholds']:
            for policy in ['final_filter', 'candidate_reselection']:
                output = []
                for slot, original in records_by_slot.items():
                    method, target = original['method'], original['chemblid']
                    items = all_candidates[slot] if policy == 'candidate_reselection' else [
                        dict(original, eligible=bool(original['delivery']), turn=0, candidate_position=0)]
                    accepted, unresolved = [], 0
                    for item in items:
                        if not item['eligible']:
                            continue
                        f = feature_map[(target, item['canonical'])]
                        value = f.get(metric)
                        if value is None and metric == 'mcs_seed_fraction':
                            if f['mcs_lower_seed_fraction'] + 1e-12 >= threshold:
                                accepted.append(item)
                            elif f['mcs_upper_seed_fraction'] + 1e-12 >= threshold:
                                unresolved += 1
                        elif value is not None and value + 1e-12 >= threshold:
                            accepted.append(item)
                    chosen = choose(accepted)
                    row = dict(method=method, chemblid=target, slot_id=slot,
                               canonical=chosen['canonical'] if chosen else None,
                               delivery=int(chosen is not None), yield_QED=chosen['QED'] if chosen else 0.)
                    if chosen:
                        row.update(feature_map[(target, chosen['canonical'])])
                    output.append(row)
                    traces.append(dict(metric=metric, threshold=threshold, policy=policy,
                        method=method, chemblid=target, slot_id=slot, original_canonical=original['canonical'],
                        selected_canonical=row['canonical'], yield_QED=row['yield_QED'],
                        delivery=row['delivery'], unresolved_candidates=unresolved))
                    if threshold == 0:
                        assert row['canonical'] == original['canonical']
                target, summary = aggregate(output)
                target_rows.extend([dict(metric=metric, threshold=threshold, policy=policy, **r) for r in target])
                summaries.extend([dict(metric=metric, threshold=threshold, policy=policy, **r) for r in summary])
    frame = pd.DataFrame(summaries)
    for _, part in frame.groupby(['metric', 'policy', 'method']):
        ordered = part.sort_values('threshold')
        assert (np.diff(ordered.delivery) <= 1e-10).all()
        assert (np.diff(ordered.yield_QED) <= 1e-10).all()
    paired_methods = frame.pivot(index=['metric', 'threshold', 'method'], columns='policy', values='yield_QED')
    assert (paired_methods.candidate_reselection + 1e-12 >= paired_methods.final_filter).all()
    return target_rows, summaries, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-mcs', action='store_true', help='Reuse only matching MCS cache entries')
    args = parser.parse_args()
    started = time.monotonic()
    (ROOT / 'results').mkdir(exist_ok=True)
    if (ROOT / 'results/summary.json').exists():
        parser.error('Results already complete; choose a fresh package to rerun')
    frozen = json.loads((ROOT / 'frozen_inputs_and_code.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT / p) == digest for p, digest in frozen.items())
    protocol = json.loads((ROOT / 'protocol.json').read_text(encoding='utf-8'))
    tasks = read('tasks25.json')
    task_map = {t['chemblid']: t for t in tasks}
    original = read('B_planned375.json')
    candidates = read('B_candidates_lossless2244.json')
    refs = read('B_references150.json')
    library = pd.read_csv(ROOT / 'inputs/library.csv')
    assert len(tasks) == 25 and len(original) == 375 and len(candidates) == 2244
    assert len(library) == 26108 and library.canonical.is_unique
    library_set = set(library.canonical)
    library_ns = {canonical(Chem.MolFromSmiles(s), False) for s in library_set}
    triplets = pd.read_csv(ROOT / 'inputs/triplets_ring_count.csv')
    training_raw = set(triplets[['anchor_smiles', 'positive_smiles', 'negative_smiles']].to_numpy().ravel())
    training_set = {canonical(Chem.MolFromSmiles(s)) for s in training_raw}
    assert training_set <= library_set
    baseline, baseline_candidates, pools, reference_matches = retrieve(tasks, library, refs)
    save_csv('retrieval_pools.csv', pools)
    save_csv('retrieval_baseline_candidates.csv', baseline_candidates)
    print('Three retrieval baselines built; original Morgan top-three validation: 75/75.', flush=True)
    by_slot = defaultdict(list)
    for row in candidates:
        by_slot[row['slot_id']].append(row)
        if row['valid']:
            properties = describe(row['smiles'])
            assert properties['canonical'] == row['canonical']
            for key in ['RC', 'QED', 'SA_raw', 'MW', 'HeavyAtoms', 'fragments']:
                assert abs(properties[key] - row[key]) < 1e-9, (row['slot_id'], key)
        else:
            mol = Chem.MolFromSmiles(row['smiles']) if row.get('smiles') else None
            assert mol is None or mol.GetNumHeavyAtoms() == 0
    for row in original:
        selected = choose(by_slot[row['slot_id']])
        assert (selected['canonical'] if selected else None) == row['canonical']
    print('Original selector identity verified at all 375 planned positions.', flush=True)
    pairs, required = {}, set()
    for row in candidates + baseline_candidates:
        if not row.get('valid', True):
            continue
        key = (row['chemblid'], row['canonical'])
        required.add(key)
        seed, molecule = describe(task_map[key[0]]['seed_smiles']), describe(key[1])
        if seed['organic'] and molecule['organic']:
            pairs[mcs_key(seed['organic'], molecule['organic'])] = (seed['organic'], molecule['organic'])
    cache = compute_mcs(pairs, protocol)
    feature_map = {key: features(task_map[key[0]], key[1], cache, library_set, library_ns, training_set)
                   for key in sorted(required)}
    feature_rows = [dict(chemblid=key[0], **value) for key, value in feature_map.items()]
    save_csv('molecule_structure_features.csv', feature_rows)
    records = []
    for source in original:
        row = dict(method=source['group'], chemblid=source['chemblid'], slot_id=source['slot_id'],
                   position=source['position'], canonical=source['canonical'],
                   delivery=int(source['exact_noncopy_success']), yield_QED=source['compliant_QED_utility'])
        if source['canonical']:
            row.update(feature_map[(source['chemblid'], source['canonical'])])
            assert abs(row['tanimoto_parent'] - source['tanimoto']) < 1e-12
            assert abs(row['QED'] - source['QED']) < 1e-12
        records.append(row)
    for source in baseline:
        row = dict(method=source['method'], chemblid=source['chemblid'], slot_id=source['slot_id'], position=1,
                   canonical=source['canonical'], delivery=1, yield_QED=source['QED'],
                   library_id=source['library_id'], retrieval_rank=source['retrieval_rank'], cosine=source['cosine'])
        row.update(feature_map[(source['chemblid'], source['canonical'])])
        records.append(row)
    target_rows, summary = aggregate(records)
    target_frame = pd.DataFrame(target_rows)
    tests = []
    for method in ['T', 'M', 'G']:
        for base in ['R_NN1', 'R_QED3', 'R_QED6']:
            a = target_frame[target_frame.method == method].set_index('chemblid').yield_QED.sort_index()
            b = target_frame[target_frame.method == base].set_index('chemblid').yield_QED.sort_index()
            assert a.index.equals(b.index)
            tests.append(dict(comparison=method + '-' + base, family='nine_exploratory_retrieval_QED_yield_tests',
                              metric='yield_QED', **paired((a - b).to_numpy())))
    correct_holm(tests)
    candidate_rows = [dict(c, method=c['group']) for c in candidates] + baseline_candidates
    gate_targets, gate_summary, trace = gated_analysis(records, candidate_rows, feature_map, tasks, protocol)
    correlations = []
    for method in METHODS:
        part = target_frame[target_frame.method == method]
        for predictor in ['tanimoto_parent', 'mcs_seed_fraction', 'dHA_organic', 'dMW_organic']:
            vals = part[[predictor, 'dQED_parent']].dropna()
            if vals[predictor].nunique() > 1 and vals.dQED_parent.nunique() > 1:
                value = float(np.corrcoef(vals[predictor].rank(), vals.dQED_parent.rank())[0, 1])
            else:
                value = None
            correlations.append(dict(method=method, predictor=predictor, outcome='dQED_parent',
                                     n_target_means=len(vals), spearman=value, interpretation='descriptive association'))
    save_csv('final_outputs_and_baselines450.csv', records)
    save_csv('target_means150.csv', target_rows)
    save_csv('method_summary6.csv', summary)
    save_csv('retrieval_paired_tests9.csv', tests)
    save_csv('structure_gate_target_means.csv', gate_targets)
    save_csv('structure_gate_summary.csv', gate_summary)
    save_csv('structure_gate_selections.csv', trace)
    save_csv('target_correlations.csv', correlations)
    for c in candidate_rows:
        if c.get('valid', True) and c.get('canonical'):
            c.update(feature_map[(c['chemblid'], c['canonical'])])
    save_csv('candidate_structure_audit.csv', candidate_rows)
    incomplete = [r for r in cache.values() if not r['complete']]
    dump(ROOT / 'results/summary.json', dict(protocol_sha256=sha(ROOT / 'protocol.json'),
        methods=summary, exploratory_tests=tests, correlations=correlations,
        MCS=dict(unique_pairs=len(pairs), complete=len(pairs) - len(incomplete), incomplete=len(incomplete),
                 incomplete_pairs=incomplete), training_raw_unique=len(training_raw),
        training_canonical_unique=len(training_set), training_is_subset_of_library=True,
        delivered_LLM=373, distinct_LLM=293, seed_atomic_counterion_cases=['CHEMBL1821'],
        molecule_feature_pairs=len(feature_rows), new_API_calls=0, new_training_jobs=0, new_docking_jobs=0))
    assert all(sha(ROOT / p) == digest for p, digest in frozen.items())
    dump(ROOT / 'results/verification.json', dict(status='passed', inputs_and_frozen_code_unchanged=True,
        original_375_selections_exact=True, original_Morgan_reference_identities_and_scores_matched=reference_matches,
        all_valid_original_candidate_descriptors_recomputed=True, original_final_QED_and_Tanimoto_match=True,
        top6_QED_ge_top3_QED_ge_nearest=True, all_zero_threshold_outputs_match_originals=True,
        all_structure_curves_monotone=True, reselected_yield_ge_final_filter_yield=True,
        paired_tests_use_25_targets=True, inference_family='nine exploratory yield tests, Holm',
        MCS_complete=len(pairs) - len(incomplete), MCS_incomplete=len(incomplete),
        elapsed_seconds=round(time.monotonic() - started, 3),
        versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, rdkit=rdBase.rdkitVersion),
        new_API_calls=0, new_training_jobs=0, new_docking_jobs=0))
    print('Offline numerical analysis complete; original science inputs unchanged.', flush=True)


if __name__ == '__main__':
    main()
