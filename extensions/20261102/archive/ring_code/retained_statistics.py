"""Unchanged released statistical and historical selector functions."""
import numpy as np

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
