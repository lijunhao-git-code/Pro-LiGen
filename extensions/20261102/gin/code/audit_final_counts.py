"""A small independent raw-record check of published aggregate counts and means."""
import csv,json,statistics
from collections import defaultdict
from common import ROOT,read,sha,now
from analysis_utils import save_json


def main():
    rows=[read(p) for p in sorted((ROOT/'results').glob('*.json'))];assert len(rows)==1000
    summary=list(csv.DictReader((ROOT/'analysis/properties/group_summary.csv').open(encoding='utf-8-sig')))
    grouped=defaultdict(list)
    for r in rows:grouped[(r['property'],r['group'])].append(r)
    checks=[]
    for record in summary:
        key=(record['property'],record['group']);part=grouped[key];assert len(part)==125
        successes=sum(r['ok'] for r in part);utility=statistics.fmean(r['selected_candidate']['QED'] if r['ok'] else 0. for r in part)
        assert int(record['delivered'])==successes
        assert abs(float(record['S'])-successes/125)<1e-13 and abs(float(record['U'])-utility)<1e-13
        tasks=defaultdict(list)
        for r in part:tasks[r['chemblid']].append(r)
        assert len(tasks)==25 and all(len(values)==5 for values in tasks.values())
        conditional=statistics.fmean(statistics.fmean(r['selected_candidate']['QED'] for r in vals if r['ok']) for vals in tasks.values() if any(r['ok'] for r in vals))
        assert abs(float(record['QED'])-conditional)<1e-13
        checks.append(dict(property=key[0],group=key[1],planned=125,delivered=successes,S=successes/125,U=utility,QED=conditional))
    positions=list(csv.DictReader((ROOT/'analysis/properties/candidate_grid.csv').open(encoding='utf-8-sig')));assert len(positions)==6000
    assert len({(r['slot_id'],r['turn'],r['candidate_position']) for r in positions})==6000
    tests=list(csv.DictReader((ROOT/'analysis/properties/declared_tests.csv').open(encoding='utf-8-sig')))
    assert len(tests)==16 and sum(r['family']=='primary8' for r in tests)==8 and sum(r['family']=='secondary8' for r in tests)==8
    save_json(ROOT/'analysis/independent_count_verification.json',dict(status='passed',time_utc=now(),raw_results=1000,
        nominal_candidates=6000,all8_group_counts_and_U_and_conditional_QED_match=True,all25x5x4x2_positions_present=True,
        primary_family8_and_secondary_family8=True,checks=checks,new_API_calls=0))
    print('INDEPENDENT COUNTS:8 groups,1000 positions,6000 proposal positions verified.',flush=True)


if __name__=='__main__':main()
