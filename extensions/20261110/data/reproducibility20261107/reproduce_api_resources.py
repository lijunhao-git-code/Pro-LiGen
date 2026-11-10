"""Verify API usage and timings offline. No model calls or source mutation."""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone,timedelta
from statistics import median
import argparse,csv,json,math,hashlib
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b):assert abs(float(a)-float(b))<1e-8,(a,b)
def percentile(v,q):
    v=sorted(v);z=(len(v)-1)*q;i=int(z);j=min(i+1,len(v)-1)
    return v[i] if i==j else v[i]*(j-z)+v[j]*(z-i)
def main(data):
    calls=rows(data/'api_calls7504.csv');runs=rows(data/'runtime_runs1500.csv');summary=rows(data/'cohort_resources.csv')
    assert len(calls)==7504 and len(runs)==1500 and len(summary)==4
    byrun=defaultdict(list)
    for c in calls:
        hit=int(c['cache_hit_tokens']);miss=int(c['prompt_tokens'])-hit;output=int(c['completion_tokens'])
        assert hit>=0 and miss>=0
        peak=(hit*.04+miss*2.+output*8.)/1e6
        dt=datetime.fromisoformat(c['started_utc']).astimezone(timezone(timedelta(hours=8)))
        ispeak=dt.weekday()<5 and (9<=dt.hour<12 or 14<=dt.hour<18)
        close(c['estimated_cny'],peak if ispeak else peak/2);close(c['all_peak_upper_cny'],peak)
        assert c['returned_model']=='deepseek-flash'
        byrun[c['slot_id']].append(c)
    assert len(byrun)==1500
    for r in runs:
        c=byrun[r['slot_id']];assert len(c)==int(r['calls'])
        assert all(q['condition']==r['condition'] and q['group']==r['group'] for q in c)
        close(r['api_active_seconds'],math.fsum(float(q['elapsed_seconds']) for q in c))
        for k in ['estimated_cny','all_peak_upper_cny']:close(r[k],math.fsum(float(q[k]) for q in c))
    for s in summary:
        part=[r for r in runs if s['condition']=='total' or r['condition']==s['condition']]
        assert len(part)==int(s['planned_runs']) and sum(int(r['calls']) for r in part)==int(s['calls'])
        for k in ['estimated_cny','all_peak_upper_cny']:close(s[k],math.fsum(float(r[k]) for r in part))
        close(s['mean_cny_per_run'],float(s['estimated_cny'])/len(part))
        vals=[float(r['api_active_seconds']) for r in part]
        close(s['median_api_active_seconds'],median(vals));close(s['p90_api_active_seconds'],percentile(vals,.9))
        close(s['q25_api_active_seconds'],percentile(vals,.25));close(s['q75_api_active_seconds'],percentile(vals,.75))
    metadata=json.loads((data/'submitted_request_metadata.json').read_text())
    assert sum(q['request_settings'][0]['calls'] for q in metadata.values())==7504
    for q in metadata.values():
        assert q['missing_request_files']==0 and len(q['request_settings'])==1
        s=q['request_settings'][0]['settings']
        assert s['model']=='deepseek-flash' and s['max_tokens']==16384 and s['reasoning_effort']=='low'
        assert s['thinking']=={'type':'enabled'} and 'temperature' not in s and 'top_p' not in s
    manifest=json.loads((data/'source_manifest.json').read_text())
    for n,h in manifest.items():assert hashlib.sha256((data/n).read_bytes()).hexdigest()==h,n
    return dict(status='passed',requests_verified=7504,runs_verified=1500,summary_rows_verified=4,
      main_estimated_cny=float(summary[-1]['estimated_cny']),all_peak_cny=float(summary[-1]['all_peak_upper_cny']),
      full_request_settings_checked=7504,API_calls_by_auditor=0)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(Path(__file__).resolve().parent));p.add_argument('--out')
    a=p.parse_args();result=main(Path(a.data).resolve())
    if a.out:
        dest=Path(a.out);assert not dest.exists(),'Choose a new output file';dest.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))
