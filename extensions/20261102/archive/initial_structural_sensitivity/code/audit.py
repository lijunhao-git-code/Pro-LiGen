"""Independent checks of pool ranking, eligibility, aggregates, gates, and figure coordinates."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    r=ROOT/'results'
    tasks=json.loads((ROOT/'inputs/tasks25.json').read_text(encoding='utf-8'))
    tasks={t['chemblid']:t for t in tasks}
    refs=json.loads((ROOT/'inputs/B_references150.json').read_text(encoding='utf-8'))
    refsets=defaultdict(set)
    for ref in refs:refsets[(ref['chemblid'],ref['group'])].add(ref['canonical'])
    cand=json.loads((ROOT/'inputs/B_candidates_lossless2244.json').read_text(encoding='utf-8'))
    feature=pd.read_csv(r/'molecule_structure_features.csv')
    fmap={(x['chemblid'],x['canonical']):x for x in feature.to_dict('records')}
    eligible_counts=defaultdict(int)
    for c in cand:
        if c['valid']:
            f=fmap[(c['chemblid'],c['canonical'])]
            expected=(f['RC']==tasks[c['chemblid']]['target_rc'] and f['fragments']==1
                      and c['canonical']!=tasks[c['chemblid']]['seed_canonical']
                      and c['canonical'] not in refsets[(c['chemblid'],c['group'])])
        else:expected=False
        assert expected==c['eligible'],c['slot_id']
        eligible_counts[c['group']]+=int(expected)
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    lib=pd.read_csv(ROOT/'inputs/library.csv')
    mols=[Chem.MolFromSmiles(s) for s in lib.canonical]
    fps=[gen.GetFingerprint(m) for m in mols]
    singles=[len(Chem.GetMolFrags(m))==1 for m in mols]
    b=pd.read_csv(r/'retrieval_baseline_candidates.csv')
    # Direct RDKit cosine, rather than the Tanimoto/popcount identity in the analysis.
    for name,t in tasks.items():
        fp=gen.GetFingerprint(Chem.MolFromSmiles(t['seed_smiles']))
        indices=[i for i in range(len(lib)) if singles[i] and lib.iloc[i].RingCount==t['target_rc']
                 and lib.iloc[i].canonical!=t['seed_canonical']]
        score={i:DataStructs.CosineSimilarity(fp,fps[i]) for i in indices}
        ranked=sorted(indices,key=lambda i:(-score[i],lib.iloc[i].canonical))[:6]
        saved=b[(b.chemblid==name)&(b.method=='R_QED6')].sort_values('retrieval_rank')
        assert saved.canonical.tolist()==[lib.iloc[i].canonical for i in ranked]
        assert np.allclose(saved.cosine,[score[i] for i in ranked],atol=1e-12,rtol=0)
    final=pd.read_csv(r/'final_outputs_and_baselines450.csv')
    summary=pd.read_csv(r/'method_summary6.csv')
    mean=pd.read_csv(r/'target_means150.csv')
    for _,s in summary.iterrows():
        part=final[final.method==s.method]
        assert len(part)==s.planned and int(part.delivery.sum())==s.delivered
        for metric in ['yield_QED','QED','tanimoto_parent','mcs_seed_fraction','dHA_organic','dMW_organic']:
            target=part.groupby('chemblid')[metric].mean()
            assert abs(target.mean()-s[metric])<1e-12
            assert np.allclose(target.sort_index(),mean[mean.method==s.method].set_index('chemblid')[metric].sort_index(),atol=1e-12,rtol=0)
    trace=pd.read_csv(r/'structure_gate_selections.csv')
    gates=pd.read_csv(r/'structure_gate_summary.csv')
    for _,s in gates.iterrows():
        part=trace[(trace.method==s.method)&(trace.metric==s.metric)&(trace.threshold==s.threshold)&(trace.policy==s.policy)]
        assert len(part)==s.planned and part.delivery.sum()==s.delivered
        assert abs(part.yield_QED.mean()-s.yield_QED)<1e-12
    coords=pd.read_csv(ROOT/'figures/plotted_coordinates.csv')
    for _,p in coords[coords.figure=='01_baseline_comparison'].iterrows():
        if p.mark=='mean':
            expected=summary.set_index('method').loc[p.method,p.panel]
        else:
            expected=mean.set_index(['method','chemblid']).loc[(p.method,p.chemblid),p.panel]
        assert abs(p.y-expected)<1e-12
    for method in ['T','M','G']:
        delivered=final[(final.method==method)&(final.delivery==1)]
        for metric,panel in [('tanimoto_parent','tanimoto_ECDF'),('mcs_seed_fraction','MCS_ECDF')]:
            part=coords[(coords.figure=='02_structural_changes')&(coords.method==method)&(coords.panel==panel)]
            expected=np.sort(delivered[metric].dropna())
            assert np.allclose(part.x,expected,atol=1e-12,rtol=0)
            assert np.allclose(part.y,np.arange(1,len(expected)+1)/len(expected),atol=1e-12,rtol=0)
    for _,p in coords[coords.figure=='03_structure_thresholds'].iterrows():
        metric,outcome=p.panel.split('__')
        s=gates[(gates.method==p.method)&(gates.metric==metric)&(gates.threshold==p.x)&(gates.policy==p.policy)]
        assert len(s)==1 and abs(s.iloc[0][outcome]-p.y)<1e-12
    bounds=[]
    for method,part in final[final.delivery==1].groupby('method'):
        lower=part.groupby('chemblid').mcs_lower_seed_fraction.mean().mean()
        upper_values=np.where(part.mcs_complete==True,part.mcs_lower_seed_fraction,part.mcs_upper_seed_fraction)
        upper=part.assign(upper=upper_values).groupby('chemblid').upper.mean().mean()
        bounds.append(dict(method=method,n_delivered=len(part),n_MCS_complete=int((part.mcs_complete==True).sum()),
                           lower_MCS_mean=lower,upper_MCS_mean=upper))
    pd.DataFrame(bounds).to_csv(r/'MCS_completion_bounds.csv',index=False,encoding='utf-8-sig')
    counts=final[final.delivery==1].groupby('method').agg(
        delivered=('slot_id','count'),qed_above_seed=('dQED_parent',lambda x:int((x>0).sum())),
        lost_at_least_10_organic_heavy_atoms=('dHA_organic',lambda x:int((x<=-10).sum())),
        same_or_more_organic_heavy_atoms=('dHA_organic',lambda x:int((x>=0).sum())))
    counts.to_csv(r/'chemical_change_counts.csv',encoding='utf-8-sig')
    status=dict(status='passed',independently_recomputed_candidate_eligibility=dict(eligible_counts),
        direct_RDKit_cosine_verified_top6_for_all_25_tasks=True,target_aggregate_recomputed=True,
        structure_trace_aggregates_recomputed=True,figure1_all_coordinates_verified=True,
        figure2_ECDF_coordinates_verified=True,figure3_all_coordinates_verified=True,
        MCS_bounds=bounds,displayed_coordinate_rows=len(coords),
        original_inputs_hashes_verified=all(digest(ROOT/p)==v for p,v in json.loads((ROOT/'frozen_inputs_and_code.json').read_text()).items()))
    (r/'independent_audit.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
