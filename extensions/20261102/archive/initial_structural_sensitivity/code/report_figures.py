"""Render standalone supplementary plots from exported numerical tables only."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METHODS = ['T', 'M', 'G', 'R_NN1', 'R_QED3', 'R_QED6']
LABELS = {'T': 'No examples', 'M': 'Morgan examples', 'G': 'GIN examples',
          'R_NN1': 'Retrieve NN1', 'R_QED3': 'Retrieve QED3', 'R_QED6': 'Retrieve QED6'}
COLORS = {'T': '#667085', 'M': '#087F8C', 'G': '#D87A22',
          'R_NN1': '#A795C3', 'R_QED3': '#8066A2', 'R_QED6': '#523775'}
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelsize': 11,
    'axes.titlesize': 12, 'svg.fonttype': 'none', 'savefig.facecolor': 'white'})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ROOT / 'figures'
    out.mkdir(exist_ok=True)
    if (out / 'figure_index.json').exists():
        raise SystemExit('Figures already exist; render into a fresh copied package.')
    source = ROOT / 'results'
    final = pd.read_csv(source / 'final_outputs_and_baselines450.csv')
    targets = pd.read_csv(source / 'target_means150.csv')
    summary = pd.read_csv(source / 'method_summary6.csv')
    gates = pd.read_csv(source / 'structure_gate_summary.csv')
    indices, points = [], []

    def save(fig, stem, sources, details):
        for ext in ['png', 'svg']:
            fig.savefig(out / (stem + '.' + ext), dpi=240, bbox_inches='tight', pad_inches=.12)
        plt.close(fig)
        indices.append(dict(figure=stem, data_files={str(p.relative_to(ROOT)): sha(p) for p in sources},
                            interpretation=details))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), layout='constrained')
    for ax, metric, title in zip(axes,
            ['yield_QED', 'tanimoto_parent', 'mcs_seed_fraction'],
            ['A  QED per planned position', 'B  Similarity to the seed', 'C  Common-atom coverage of seed']):
        for i, method in enumerate(METHODS):
            part = targets[targets.method == method].sort_values('chemblid')
            vals = part[metric].to_numpy()
            offsets = (np.arange(len(part)) - 12) / 12 * .18
            ax.scatter(i + offsets, vals, s=17, c=COLORS[method], alpha=.45, linewidths=0)
            mean = vals.mean()
            expected = summary.set_index('method').loc[method, metric]
            assert abs(mean - expected) < 1e-12
            ax.scatter(i, mean, marker='D', c=COLORS[method], edgecolors='black', linewidths=.6, s=58, zorder=3)
            ax.annotate(f'{mean:.3f}', (i, 1.025), ha='center', fontsize=9,
                        xycoords=('data', 'axes fraction'), color=COLORS[method])
            for offset, (_, row) in zip(offsets, part.iterrows()):
                points.append(dict(figure='01_baseline_comparison', panel=metric, method=method,
                                   chemblid=row.chemblid, x=i+offset, y=row[metric], mark='target'))
            points.append(dict(figure='01_baseline_comparison', panel=metric, method=method,
                               chemblid=None, x=i, y=mean, mark='mean'))
        ax.set_xticks(range(6), [LABELS[m] for m in METHODS], rotation=35, ha='right')
        ax.set_ylim(-.02, 1.02)
        ax.set_title(title, pad=28, loc='left')
        ax.set_ylabel({'yield_QED': 'QED yield', 'tanimoto_parent': 'Morgan Tanimoto',
                       'mcs_seed_fraction': 'MCS atoms / seed organic heavy atoms'}[metric])
        ax.axvline(2.5, color='#CBD5E1', linewidth=1, linestyle='--')
        ax.grid(axis='y', color='#E5E7EB', linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle('Generation and direct retrieval on the same 25 tasks', fontsize=14)
    save(fig, '01_baseline_comparison', [source/'target_means150.csv', source/'method_summary6.csv'],
         'Dots: 25 target means; diamonds: equally weighted mean. Generation has five planned positions per target, retrieval one deterministic output. No docking claim.')

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained')
    for method in ['T', 'M', 'G']:
        part = targets[targets.method == method].sort_values('chemblid')
        for ax, metric, xlabel, panel in [
            (axes[0,0], 'dHA_organic', 'Change in organic heavy atoms', 'heavy_atoms'),
            (axes[0,1], 'dMW_organic', 'Change in organic molecular weight (Da)', 'MW')]:
            ax.scatter(part[metric], part.dQED_parent, s=29, color=COLORS[method], alpha=.7,
                       label=LABELS[method], linewidths=.3, edgecolors='white')
            for _, row in part.iterrows():
                points.append(dict(figure='02_structural_changes', panel=panel, method=method,
                                   chemblid=row.chemblid, x=row[metric], y=row.dQED_parent, mark='target'))
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Change in parent-form QED')
        delivered = final[(final.method == method) & (final.delivery == 1)]
        for ax, metric, label, panel in [
            (axes[1,0], 'tanimoto_parent', 'Morgan Tanimoto to seed', 'tanimoto_ECDF'),
            (axes[1,1], 'mcs_seed_fraction', 'MCS atoms / seed organic heavy atoms', 'MCS_ECDF')]:
            vals = np.sort(delivered[metric].dropna().to_numpy())
            ys = np.arange(1, len(vals)+1) / len(vals)
            ax.step(vals, ys, where='post', color=COLORS[method], lw=2,
                    label=f'{LABELS[method]} (n={len(vals)})')
            for x, y in zip(vals, ys):
                points.append(dict(figure='02_structural_changes', panel=panel, method=method,
                                   chemblid=None, x=x, y=y, mark='delivered_output'))
            ax.set_xlabel(label)
            ax.set_ylabel('Fraction of delivered outputs at or below x')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
    for ax in axes[0]:
        ax.axhline(0, color='#94A3B8', lw=.8)
        ax.axvline(0, color='#94A3B8', lw=.8)
    for ax, title in zip(axes.ravel(), [
            'A  Heavy-atom change and QED improvement', 'B  Molecular-weight change and QED improvement',
            'C  Final-output fingerprint similarity', 'D  Final-output common-atom coverage']):
        ax.set_title(title, loc='left', fontsize=11)
        ax.grid(color='#E5E7EB', lw=.6)
        ax.set_axisbelow(True)
    axes[0,0].legend(loc='best', fontsize=8)
    axes[1,1].legend(loc='upper left', fontsize=8)
    save(fig, '02_structural_changes', [source/'target_means150.csv', source/'final_outputs_and_baselines450.csv'],
         'Top: 25 target means per generation group, associations only. Bottom: descriptive ECDF of delivered outputs (123/125/125), no independent-output statistical test. MCS permits ring-to-chain matching.')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout='constrained')
    for j, (metric, xlabel) in enumerate([
            ('tanimoto_parent', 'Minimum Morgan Tanimoto to seed'),
            ('mcs_seed_fraction', 'Minimum MCS coverage of seed')]):
        for i, (outcome, ylabel) in enumerate([('delivery', 'Accepted / planned positions'),
                                              ('yield_QED', 'QED per planned position')]):
            ax = axes[i,j]
            for method in ['T', 'M', 'G', 'R_QED6']:
                for policy, ls in [('final_filter', '--'), ('candidate_reselection', '-')]:
                    part = gates[(gates.metric == metric) & (gates.method == method) & (gates.policy == policy)].sort_values('threshold')
                    assert len(part) == 10
                    ax.plot(part.threshold, part[outcome], linestyle=ls, lw=1.8, color=COLORS[method])
                    for _, row in part.iterrows():
                        points.append(dict(figure='03_structure_thresholds', panel=metric+'__'+outcome,
                            method=method, policy=policy, chemblid=None, x=row.threshold, y=row[outcome], mark='mean'))
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_xlim(0, .9)
            ax.set_ylim(-.015, 1.03)
            ax.grid(color='#E5E7EB', lw=.6)
            ax.set_axisbelow(True)
    for ax, title in zip(axes.ravel(), ['A  Fingerprint condition: acceptance', 'B  Common-atom condition: acceptance',
                                       'C  Fingerprint condition: QED yield', 'D  Common-atom condition: QED yield']):
        ax.set_title(title, loc='left', fontsize=11)
    handles = [Line2D([0], [0], color=COLORS[m], lw=2, label=LABELS[m]) for m in ['T','M','G','R_QED6']]
    handles += [Line2D([0], [0], color='#475569', ls='--', lw=1.7, label='Filter original final output'),
                Line2D([0], [0], color='#475569', ls='-', lw=1.7, label='Reselect from saved candidates')]
    fig.legend(handles=handles, ncol=3, loc='outside lower center', fontsize=9, frameon=False)
    save(fig, '03_structure_thresholds', [source/'structure_gate_summary.csv'],
         'All thresholds 0.0-0.9. Solid reselects from existing candidates, dashed filters original final. Every failure remains in planned denominator. Exploratory sensitivity, not new generation.')
    pd.DataFrame(points).to_csv(out/'plotted_coordinates.csv', index=False, encoding='utf-8-sig')
    (out/'figure_index.json').write_text(json.dumps(indices, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('Three figures rendered (PNG + editable SVG) with source hashes and plotted coordinates.')


if __name__ == '__main__':
    main()
