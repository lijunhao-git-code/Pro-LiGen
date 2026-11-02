"""Independently audit all16 pilot outputs and draw their real molecular structures."""
import json,sqlite3
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import Draw,rdFingerprintGenerator
from common import ROOT,PROPERTIES,properties,read,read_jsonl,dump,sha,now


def main():
    RDLogger.DisableLog('rdApp.*')
    schedule=[r for r in read(ROOT/'inputs/schedule1000.json') if r['pilot']]
    entries={r['slot_id']:r for r in read_jsonl(ROOT/'inputs/entries1000.jsonl')}
    absent=[r['slot_id'] for r in schedule if not (ROOT/'results'/f'{r["slot_id"]}.json').exists()]
    if absent:
        dump(ROOT/'analysis/pilot_audit.json',dict(status='incomplete',missing=absent,full_run_allowed=False,time_utc=now()))
        print(json.dumps(dict(status='incomplete',missing_count=len(absent))));return
    fp=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)
    rows=[];candidate_rows=[];hashes={};selected={};library=set(pd.read_csv(ROOT/'inputs/library.csv').canonical)
    for scheduled in schedule:
        sid=scheduled['slot_id'];entry=entries[sid];path=ROOT/'results'/f'{sid}.json';result=read(path)
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        seed=properties(entry['seed_smiles'])
        assert abs(entry['target_center']-(seed[entry['property']]+PROPERTIES[entry['property']]['delta']))<1e-9
        assert entry['tolerance']==PROPERTIES[entry['property']]['tolerance']
        assert all(abs(properties(s)[entry['property']]-entry['target_center'])<=entry['tolerance']+1e-8 for s in entry['exemplars'])
        eligible={};seedfp=fp.GetFingerprint(Chem.MolFromSmiles(entry['seed_smiles']))
        for c in result['candidates']:
            if c['valid']:
                actual=properties(c['smiles'])
                assert actual['canonical']==c['canonical']
                expected=(actual['fragments']==1 and actual['contains_carbon']
                    and actual['canonical']!=entry['seed_smiles'] and actual['canonical'] not in entry['exemplars']
                    and abs(actual[entry['property']]-entry['target_center'])<=entry['tolerance']+1e-8)
                assert expected==c['eligible']
                assert all(abs(actual[k]-c[k])<1e-9 for k in ['logp','tpsa','QED','SA_raw','MW','HeavyAtoms','RC'])
                if expected:eligible[c['canonical']]=c
            else:
                mol=Chem.MolFromSmiles(c['smiles']) if c.get('smiles') else None
                assert mol is None or mol.GetNumHeavyAtoms()==0 or c['reason']=='missing_tagged_proposal'
            candidate_rows.append(dict(slot_id=sid,property=entry['property'],chemblid=entry['chemblid'],group=entry['group'],**c))
        chosen=min(eligible.values(),key=lambda c:(-c['QED'],c['SA_raw'],c['canonical'])) if eligible else None
        assert (chosen['canonical'] if chosen else None)==result['canonical']
        row=dict(slot_id=sid,property=entry['property'],chemblid=entry['chemblid'],group=entry['group'],
            technical_complete=result['technical_complete'],delivery=int(result['ok']),canonical=result['canonical'],seed_smiles=entry['seed_smiles'],
            seed_property=entry['seed_property'],target_center=entry['target_center'],target_lower=entry['target_lower'],target_upper=entry['target_upper'],
            valid_proposals=sum(c['valid'] for c in result['candidates']),eligible_proposals=sum(c['eligible'] for c in result['candidates']),
            eligible_unique=len(eligible),QED_yield=chosen['QED'] if chosen else 0.)
        if chosen:
            row.update(property_value=chosen[entry['property']],delta_property=chosen[entry['property']]-entry['seed_property'],
                target_error=chosen[entry['property']]-entry['target_center'],QED=chosen['QED'],SA_raw=chosen['SA_raw'],
                tanimoto=float(DataStructs.TanimotoSimilarity(seedfp,fp.GetFingerprint(Chem.MolFromSmiles(chosen['canonical'])))),
                library_exact_match=chosen['canonical'] in library,delta_MW=chosen['MW']-seed['MW'],delta_HeavyAtoms=chosen['HeavyAtoms']-seed['HeavyAtoms'])
        selected[(entry['property'],entry['chemblid'],entry['group'])]=row;rows.append(row)
    slots=pd.DataFrame(rows);candidates=pd.DataFrame(candidate_rows)
    slots.to_csv(ROOT/'analysis/pilot_slots16.csv',index=False,encoding='utf-8-sig')
    candidates.to_csv(ROOT/'analysis/pilot_candidates96.csv',index=False,encoding='utf-8-sig')
    summary=[]
    for (prop,group),part in slots.groupby(['property','group']):
        summary.append(dict(property=prop,group=group,planned=len(part),delivered=int(part.delivery.sum()),
            valid_candidates=int(part.valid_proposals.sum()),eligible_candidates=int(part.eligible_proposals.sum()),
            QED_yield=float(part.QED_yield.mean()),QED=float(part.QED.mean()) if 'QED' in part and part.QED.notna().any() else None))
    # Real source SMILES only; empty cells explicitly mark no output.
    tasks=read(ROOT/'inputs/tasks25.json');mols=[];legends=[]
    for prop in PROPERTIES:
        for task in tasks[:2]:
            value=task[prop];mols.append(Chem.MolFromSmiles(task['seed_smiles']))
            legends.append(f'{prop.upper()} seed {task["chemblid"]}\n{value:.3f}')
            for group in ['T','M','G','R']:
                record=selected[(prop,task['chemblid'],group)]
                mols.append(Chem.MolFromSmiles(record['canonical']) if record['canonical'] else Chem.Mol())
                legends.append(f'{group}: {record["property_value"]:.3f}\nQED {record["QED"]:.3f}' if record['canonical'] else f'{group}: NO QUALIFIED OUTPUT')
    Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(300,260),legends=legends).save(ROOT/'analysis/pilot_structures.png')
    svg=Draw.MolsToGridImage(mols,molsPerRow=5,subImgSize=(300,260),legends=legends,useSVG=True)
    (ROOT/'analysis/pilot_structures.svg').write_text(svg,encoding='utf-8')
    report=dict(status='checks_passed_pending_manual_review',time_utc=now(),planned=16,completed=len(rows),
        technical_complete=int(slots.technical_complete.sum()),delivered=int(slots.delivery.sum()),
        candidates=len(candidate_rows),all_descriptor_and_eligibility_recomputations_match=True,
        all_reference_conditions_match=True,all_final_selection_identities_match=True,
        per_property_group=summary,pilot_output_hashes=hashes,
        full_run_allowed=False,note='Candidate and structure inspection required; this file does not open the full-run gate. No GIN superiority criterion.')
    dump(ROOT/'analysis/pilot_audit.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
