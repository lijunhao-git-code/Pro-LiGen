"""Portable, offline reanalysis of the supplied scientific records.

This does NOT call LLMs, train encoders, rerun docking, or alter paper inputs.
Only a new explicitly chosen output directory is written.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/SI_records'
ATOL = 1e-10


def read(name):
    return json.loads((DATA / name).read_text(encoding='utf-8'))


def frame(name):
    return pd.DataFrame(read(name))


def close(actual, expected):
    assert np.isclose(actual, expected, atol=ATOL, rtol=0), (actual, expected)


def interval(values):
    values = np.asarray(values, dtype=float)
    assert values.size and np.isfinite(values).all()
    indices = np.random.default_rng(42).integers(0, len(values), (10000, len(values)))
    bounds = np.quantile(values[indices].mean(axis=1), [.025, .975])
    return dict(difference=float(values.mean()), ci_low=float(bounds[0]), ci_high=float(bounds[1]))


def paired(values):
    values = np.asarray(values, dtype=float)
    assert len(values) == 25, 'Inference requires 25 target means, not independent positions'
    result = interval(values)
    signs = np.random.default_rng(42).choice([-1., 1.], (100000, len(values)))
    extreme = np.abs((signs * values).mean(axis=1)) >= abs(values.mean()) - 1e-12
    result.update(p_raw=float((1 + extreme.sum()) / 100001), n_targets=25)
    return result


def correct_holm(rows):
    for family in sorted({r['family'] for r in rows}):
        part = sorted([r for r in rows if r['family'] == family], key=lambda r: r['p_raw'])
        previous = 0.
        for i, row in enumerate(part):
            previous = max(previous, min(1., row['p_raw'] * (len(part) - i)))
            row['p_holm'] = previous


def choose(candidates, policy='union_QED'):
    items = sorted(candidates, key=lambda r: (r['turn'], r['candidate_position']))
    assert all(c['turn'] in [4, 5] and c['candidate_position'] in [1, 2, 3] for c in items)
    assert len({(c['turn'], c['candidate_position']) for c in items}) == len(items)
    if policy == 'turn4_QED':
        items = [c for c in items if c['turn'] == 4]
    elif policy == 'turn5_QED':
        items = [c for c in items if c['turn'] == 5]
    eligible = [c for c in items if c['eligible']]
    if not eligible:
        return None
    if policy == 'union_first_eligible':
        return eligible[0]
    # Match deployed canonical deduplication: last occurrence survives.
    eligible = list({c['canonical']: c for c in eligible}.values())
    return min(eligible, key=lambda c: (-c['QED'], c['SA_raw'], c['canonical']))


def historical():
    a = frame('A_retained500.json')
    assert len(a) == 500 and a.chemblid.nunique() == 25
    assert a.groupby(['group', 'chemblid']).size().eq(5).all()
    fields = ['dRC_seed_minus_output', 'exact_one_less_ring_diagnostic', 'QED',
              'SA_standard_raw', 'sim', 'exact_single_noncopy_diagnostic']
    means = a.groupby(['chemblid', 'group'])[fields].mean().groupby('group').mean()
    return [dict(group=g, retained=125, targets=25,
                 **{k: float(means.loc[g, k]) for k in fields},
                 multi_fragment_n=int((a[a.group == g].fragments > 1).sum()),
                 exemplar_copy_n=int(a[a.group == g].canonical_property_exemplar_copy.sum()))
            for g in ['NG', 'TG', 'PG-aggr', 'PG-mild']]


def current():
    b = frame('B_planned375.json')
    assert len(b) == 375 and not b.slot_id.duplicated().any()
    assert b.groupby(['chemblid', 'group']).size().eq(5).all()
    missing = b.exact_noncopy_success == 0
    assert b.loc[missing, ['QED', 'SA_raw']].isna().all().all()
    assert b.loc[missing, 'compliant_QED_utility'].eq(0).all()
    means = b.groupby(['chemblid', 'group']).agg(
        S=('exact_noncopy_success', 'mean'), U=('compliant_QED_utility', 'mean'),
        QED=('QED', 'mean'), SA_raw=('SA_raw', 'mean'), tanimoto=('tanimoto', 'mean'))
    rows = []
    for g in ['T', 'M', 'G']:
        values = means.xs(g, level='group').mean()
        part = b[b.group == g]
        rows.append(dict(group=g, planned=125, delivered=int(part.exact_noncopy_success.sum()),
                         distinct=int(part.canonical.dropna().nunique()),
                         **{k: float(values[k]) for k in means.columns}))
    tests = []
    for comparison, family in [('G-M', 'primary'), ('G-T', 'secondary_delivery'), ('M-T', 'secondary_delivery')]:
        left, right = comparison.split('-')
        for metric in ['S', 'U']:
            targets = means[metric].unstack('group')
            tests.append(dict(comparison=comparison, family=family, metric=metric,
                              **paired((targets[left] - targets[right]).to_numpy())))
    correct_holm(tests)
    return dict(groups=rows, statistics=tests, global_distinct=int(b.canonical.dropna().nunique()))


def feedback():
    c = frame('C_branches150.json')
    assert len(c) == 150 and not c.slot_id.duplicated().any()
    assert c.groupby(['chemblid', 'branch']).size().eq(3).all()
    assert c.technical_complete.all()
    assert c.loc[~c.ok, ['conditional_QED', 'conditional_SA']].isna().all().all()
    assert c.loc[~c.ok, 'yield_QED'].eq(0).all()
    means = c.groupby(['chemblid', 'branch'])[['ok', 'yield_QED', 'conditional_QED', 'conditional_SA']].mean()
    rows = []
    for branch in ['Feedback', 'NoExtraFeedback']:
        part = c[c.branch == branch]
        v = means.xs(branch, level='branch').mean()
        rows.append(dict(branch=branch, planned=75, delivered=int(part.ok.sum()),
                         S=float(v['ok']), U=float(v.yield_QED), QED=float(v.conditional_QED),
                         SA_raw=float(v.conditional_SA),
                         selected_also_available_fourth=int(part.selected_also_available_fourth.sum())))
    targets = means.yield_QED.unstack('branch')
    return dict(groups=rows, primary=paired((targets.Feedback - targets.NoExtraFeedback).to_numpy()),
                global_distinct=int(c.canonical.dropna().nunique()))


def replay_and_selection():
    b = read('B_planned375.json')
    candidates = json.loads(Path(__file__).with_name('B_candidates_lossless2244.json').read_text(encoding='utf-8'))
    by_slot = {}
    for candidate in candidates:
        by_slot.setdefault(candidate['slot_id'], []).append(candidate)
    replay = read('B_fixed_candidate_replay1500.json')
    saved = {(r['slot_id'], r['policy']): r for r in replay}
    for row in b:
        for policy in ['turn4_QED', 'turn5_QED', 'union_first_eligible', 'union_QED']:
            selected = choose(by_slot.get(row['slot_id'], []), policy)
            stored = saved[(row['slot_id'], policy)]
            assert bool(stored['success']) == (selected is not None)
            assert stored['selected_canonical'] == (selected['canonical'] if selected else None)
            close(stored['compliant_QED_utility'], selected['QED'] if selected else 0.)
        deployed = choose(by_slot.get(row['slot_id'], []))
        assert row['canonical'] == (deployed['canonical'] if deployed else None)
    c = {r['slot_id']: r for r in read('C_branches150.json')}
    for branch in read('C_completed_branches150_with_candidates.json'):
        selected = choose(branch['candidates'])
        row = c[branch['slot_id']]
        assert row['ok'] == (selected is not None)
        assert row['canonical'] == (selected['canonical'] if selected else None)
        close(row['yield_QED'], selected['QED'] if selected else 0.)
    replay_frame = pd.DataFrame(replay)
    rows = []
    for policy in ['turn4_QED', 'turn5_QED', 'union_first_eligible', 'union_QED']:
        row = dict(policy=policy)
        for g in ['T', 'M', 'G']:
            part = replay_frame[(replay_frame.policy == policy) & (replay_frame.group == g)]
            utility = part.groupby('chemblid').compliant_QED_utility.mean().mean()
            row[g] = dict(planned=125, delivered=int(part.success.sum()), U=float(utility))
        rows.append(row)
    return rows


def docking():
    d = frame('docking_observations900.json')
    assert len(d) == 900 and not d.observation_id.duplicated().any()
    assert d.loc[d.docking_status != 'ok', 'vina'].isna().all()
    seed = d[(d.kind == 'seed') & (d.docking_status == 'ok')].set_index('chemblid')
    assert len(seed) == 11
    outputs = d[d.kind == 'generated']
    rows = []
    for study, groups in [('A', ['NG', 'TG', 'PG-aggr', 'PG-mild']), ('B', ['T', 'M', 'G'])]:
        for g in groups:
            original = outputs[(outputs.study == study) & (outputs.group == g)]
            part = original[original.docking_status == 'ok'].copy()
            assert set(part.chemblid) == set(seed.index)
            part['delta'] = part.vina - part.chemblid.map(seed.vina)
            part['improved'] = (part.delta < 0).astype(float)
            part['dqed'] = part.active_QED - part.chemblid.map(seed.active_QED)
            assert np.allclose(part.delta, part.dVina_output_minus_seed, atol=ATOL, rtol=0)
            targets = part.groupby('chemblid')[['vina', 'delta', 'improved', 'dqed']].mean()
            ci = interval(targets.delta.to_numpy())
            rows.append(dict(study=study, group=g, planned_or_retained=len(original), docked=len(part),
                             targets=11, Vina=float(targets.vina.mean()), delta_Vina=ci['difference'],
                             ci_low=ci['ci_low'], ci_high=ci['ci_high'],
                             improved_fraction=float(targets.improved.mean()), delta_active_QED=float(targets.dqed.mean()),
                             excluded=int(original.docking_status.str.startswith('excluded').sum()),
                             no_output=int(original.parent_smiles.isna().sum())))
    return dict(groups=rows, all_observations=len(d), all_scored=int((d.docking_status == 'ok').sum()),
                all_excluded=int(d.docking_status.str.startswith('excluded').sum()))


def verify_expected(actual, expected):
    for section, identity in [('A', 'group'), ('replay', 'policy')]:
        indexed = {r[identity]: r for r in expected[section]}
        for row in actual[section]:
            ref = indexed[row[identity]]
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    close(value, ref[key])
                elif isinstance(value, dict):
                    for k, v in value.items():
                        close(v, ref[key][k])
    for section, identity in [('B', 'group'), ('C', 'branch')]:
        indexed = {r[identity]: r for r in expected[section]['groups']}
        for row in actual[section]['groups']:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    close(value, indexed[row[identity]][key])
        assert actual[section]['global_distinct'] == expected[section]['global_distinct']
    tests = {(r['comparison'], r['metric']): r for r in expected['B']['statistics']}
    for test in actual['B']['statistics']:
        ref = tests[(test['comparison'], test['metric'])]
        assert test['family'] == ref['family']
        for key in ['difference', 'ci_low', 'ci_high', 'p_raw', 'p_holm']:
            close(test[key], ref[key])
    for key in ['difference', 'ci_low', 'ci_high', 'p_raw']:
        close(actual['C']['primary'][key], expected['C']['primary'][key])
    dock = {(r['study'], r['group']): r for r in expected['docking']['groups']}
    for row in actual['docking']['groups']:
        for key, value in row.items():
            if isinstance(value, (int, float)):
                close(value, dock[(row['study'], row['group'])][key])
    for key in ['all_observations', 'all_scored', 'all_excluded']:
        close(actual['docking'][key], expected['docking'][key])


def chemistry():
    from rdkit import Chem, rdBase
    from rdkit.Chem import QED
    from rdkit.Contrib.SA_Score import sascorer
    checked = 0
    for filename, smiles, sa in [('A_retained500.json', 'smiles', 'SA_standard_raw'),
                                 ('B_planned375.json', 'smiles', 'SA_raw'),
                                 ('B_references150.json', 'smiles', 'SA_raw')]:
        for row in read(filename):
            if not row.get(smiles):
                continue
            mol = Chem.MolFromSmiles(row[smiles])
            assert mol is not None
            assert Chem.MolToSmiles(mol, isomericSmiles=True) == row['canonical']
            close(mol.GetRingInfo().NumRings(), row['RC'])
            close(QED.qed(mol), row['QED'])
            close(sascorer.calculateScore(mol), row[sa])
            checked += 1
    return dict(rdkit=rdBase.rdkitVersion, checked_parent_records=checked,
                scope='A outputs, B delivered outputs and B references; not all protocol provenance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--chemistry', action='store_true', help='Also recompute RC/QED/standard SA with RDKit')
    args = parser.parse_args()
    destination = args.out.resolve()
    assert not destination.exists(), 'Use a NEW output directory; existing results are never overwritten'
    protected = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob('*') if p.is_file()}
    expected_path = Path(__file__).with_name('expected_frozen_results.json')
    for extra in [expected_path, Path(__file__).with_name('B_candidates_lossless2244.json')]:
        protected[extra] = hashlib.sha256(extra.read_bytes()).hexdigest()
    expected = json.loads(expected_path.read_text(encoding='utf-8'))
    actual = dict(A=historical(), B=current(), C=feedback(), replay=replay_and_selection(), docking=docking())
    verify_expected(actual, expected)
    chemistry_report = chemistry() if args.chemistry else dict(executed=False, reason='Optional --chemistry not requested')
    references = frame('B_references150.json')
    rings = references.groupby(['RC', 'group']).size().unstack('group')
    assert rings.M.equals(rings.G) and references.groupby(['chemblid', 'group']).size().eq(3).all()
    for p, digest in protected.items():
        assert hashlib.sha256(p.read_bytes()).hexdigest() == digest
    destination.mkdir(parents=True, exist_ok=False)
    (destination / 'recomputed_results.json').write_text(json.dumps(actual, ensure_ascii=False, indent=2), encoding='utf-8')
    report = dict(status='passed', inputs_unchanged=True, frozen_result_tolerance=ATOL,
                  checks=['All main table summaries', 'Original paired primary and auxiliary test families',
                          '375 B final selections', '150 C final selections', '1500 fixed-candidate replay selections',
                          'Docking coverage and seven group summaries', 'Reference ring-count control'],
                  input_sha256={p.name: digest for p, digest in protected.items()}, chemistry=chemistry_report,
                  versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                  new_API_calls=0, new_training_jobs=0, new_docking_jobs=0,
                  external_model_or_GIN_training_regenerated=False, input_distribution='public_research_release')
    (destination / 'verification.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Offline scientific reanalysis passed; records unchanged, no API/training/docking calls.')


if __name__ == '__main__':
    main()
