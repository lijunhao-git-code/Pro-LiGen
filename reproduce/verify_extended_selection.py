"""Verify exact deployed final choices from lossless scientific-field exports."""
from pathlib import Path
import argparse,csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1];X=ROOT/'extensions/20261102'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();dest=Path(a.out).resolve();assert not dest.exists();dest.mkdir(parents=True)
    final={r['slot_id']:r for f in ['ring/final_positions.csv','properties/final_positions.csv'] for r in rows(X/'data/updated20261102'/f)}
    checked=0;delivered=0
    for f in ['ring_completed.json','properties_completed.json']:
        data=json.loads((X/'lossless'/f).read_text(encoding='utf-8'))
        for r in data:
            assert r['technical_complete'] and len(r['candidates'])==6
            eligible=[c for c in sorted(r['candidates'],key=lambda x:(x['turn'],x['candidate_position'])) if c['eligible']]
            unique=list({c['canonical']:c for c in eligible}.values())
            chosen=min(unique,key=lambda c:(-c['QED'],c['SA_raw'],c['canonical'])) if unique else None
            assert chosen==r['selected_candidate'];canonical=chosen['canonical'] if chosen else None
            assert canonical==r['canonical']==(final[r['slot_id']]['canonical'] or None)
            if chosen:assert abs(chosen['QED']-float(final[r['slot_id']]['U']))<2e-11
            else:assert float(final[r['slot_id']]['U'])==0
            checked+=1;delivered+=int(chosen is not None)
    assert checked==1500 and delivered==1469
    proof=dict(status='passed',runs=checked,qualified=delivered,exact_selected_candidate_dictionaries=True,canonical_identities_match=True,API_calls=0)
    (dest/'verification.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf-8');print(json.dumps(proof))
if __name__=='__main__':main()
