"""Independent replay arithmetic checks using original archived records."""
from collections import defaultdict
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]


def main():
    inp=ROOT/'inputs';r=ROOT/'results'
    raw=json.loads((inp/'B_candidates_lossless2244.json').read_text(encoding='utf-8'))
    slots=json.loads((inp/'B_planned375.json').read_text(encoding='utf-8'))
    old=json.loads((inp/'B_fixed_candidate_replay1500.json').read_text(encoding='utf-8'))
    byslot=defaultdict(list)
    for c in raw:byslot[c['slot_id']].append(c)
    output=pd.read_csv(r/'selection_by_slot12000.csv').set_index(['slot_id','policy','k'])
    checked=0
    for entry in slots:
        rows=sorted(byslot[entry['slot_id']],key=lambda x:(x['turn'],x['candidate_position']))
        for k in range(1,7):
            eligible=[c for c in rows if c['eligible'] and (c['turn']-4)*3+c['candidate_position']<=k]
            unique=list({c['canonical']:c for c in eligible}.values())
            mean_unique=sum(c['QED'] for c in unique)/len(unique) if unique else 0.
            mean_positions=sum(c['QED'] for c in eligible)/len(eligible) if eligible else 0.
            best=max((c['QED'] for c in unique),default=0.)
            for policy,expected in [('random_unique_expected',mean_unique),('random_positions_expected',mean_positions),
                                   ('first_eligible',eligible[0]['QED'] if eligible else 0.),('best_QED',best)]:
                row=output.loc[(entry['slot_id'],policy,k)]
                assert abs(row.yield_QED-expected)<1e-12
                assert row.delivery==bool(eligible)
                checked+=1
            draw=output.loc[(entry['slot_id'],'random_unique_draw42',k)]
            assert (draw.selected_canonical in {x['canonical'] for x in unique}) if unique else pd.isna(draw.selected_canonical)
    maps={'turn4_QED':('round1_best_QED',3),'turn5_QED':('round2_best_QED',3),
          'union_first_eligible':('first_eligible',6),'union_QED':('best_QED',6)}
    for row in old:
        policy,k=maps[row['policy']];new=output.loc[(row['slot_id'],policy,k)]
        assert (None if pd.isna(new.selected_canonical) else new.selected_canonical)==row['selected_canonical']
        assert abs(new.yield_QED-row['compliant_QED_utility'])<1e-12
    for line in (r/'random_unique_support2250.jsonl').read_text(encoding='utf-8').splitlines():
        support=json.loads(line);n=len(support['canonical'])
        assert n==len(set(support['canonical']))
        if n:
            assert abs(n*support['probability_each']-1)<1e-12
            expected=sum(x*support['probability_each'] for x in support['QED'])
        else:expected=0.
        assert abs(expected-output.loc[(support['slot_id'],'random_unique_expected',support['k'])].yield_QED)<1e-12
    flat=output.reset_index();summary=pd.read_csv(r/'policy_summary.csv')
    for s in summary.itertuples():
        sub=flat[(flat.group==s.group)&(flat.policy==s.policy)&(flat.k==s.k)]
        assert len(sub)==125 and sub.delivery.sum()==s.delivered
        assert abs(sub.groupby('chemblid').yield_QED.mean().mean()-s.yield_QED)<1e-12
    grid=pd.read_csv(r/'candidate_grid2250.csv');cq=pd.read_csv(r/'candidate_target_quality75.csv')
    for row in cq.itertuples():
        sub=grid[(grid.group==row.group)&(grid.chemblid==row.chemblid)]
        assert len(sub)==30
        expected=sub.loc[sub.valid==True,'QED'].sum()/30
        assert abs(expected-row.valid_proposal_QED_yield)<1e-12
    coords=pd.read_csv(ROOT/'figures/plotted_coordinates.csv')
    tm=pd.read_csv(r/'policy_target_means.csv')
    for row in coords.itertuples():
        if row.figure=='01_selection_policies':
            sub=tm[(tm.group==row.group)&(tm.policy==row.policy)&(tm.k==6)]
            assert abs(sub[row.panel].mean()-row.y)<1e-12
        elif row.figure=='02_candidate_quality' and row.mark=='target_mean':
            value=cq.set_index(['group','chemblid']).loc[(row.group,row.chemblid),row.panel]
            assert abs(value-row.y)<1e-12
        elif row.figure=='03_candidate_count':
            policy='best_QED' if row.panel=='delivery' else row.policy
            sub=summary[(summary.group==row.group)&(summary.policy==policy)&(summary.k==row.k)]
            assert len(sub)==1 and abs(sub.iloc[0][row.panel]-row.y)<1e-12
    for g in ['T','M','G']:
        for flag in ['valid','eligible']:
            sub=coords[(coords.figure=='02_candidate_quality')&(coords.group==g)&(coords.panel==flag+'_ECDF')]
            vals=np.sort(grid[(grid.group==g)&(grid[flag]==True)].QED.to_numpy())
            assert np.allclose(sub.x,vals,atol=1e-12,rtol=0)
            assert np.allclose(sub.y,np.arange(1,len(vals)+1)/len(vals),atol=1e-12,rtol=0)
    frozen=json.loads((ROOT/'frozen_inputs_and_code.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==v for name,v in frozen.items())
    result=dict(status='passed',independent_policy_expectations_checked=checked,
        all_1500_historical_replay_records_match_new_results=True,all_2250_random_distributions_normalized=True,
        fixed_seed_random_draws_belong_to_support=True,planned_denominators_and_target_means_verified=True,
        candidate_zero_filled_utilities_verified=True,all_figure_mean_and_ECDF_coordinates_verified=True,
        frozen_inputs_and_core_code_unchanged=True,figure_coordinate_rows=len(coords))
    (r/'independent_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
