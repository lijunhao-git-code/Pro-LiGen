"""Wait for generation, then independently replay, plot, archive; no API generation."""
from pathlib import Path
import hashlib,json,os,sqlite3,subprocess,time,zipfile
ROOT=Path('/root/ProLiGen_ring_random_control_20261102')
PYTHON='/root/miniconda3/envs/ljh/bin/python'
read=lambda p:json.loads(p.read_text())
def dump(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',
    LD_PRELOAD='/root/miniconda3/envs/ljh/lib/libstdc++.so.6',LD_LIBRARY_PATH='/root/miniconda3/envs/ljh/lib:'+env.get('LD_LIBRARY_PATH',''))
def run(args):
    p=subprocess.run([PYTHON,*args],cwd=ROOT,env=env,capture_output=True,text=True)
    print(p.stdout[-12000:],flush=True)
    if p.returncode:raise RuntimeError(p.stderr[-6000:])
    return p
def main():
    dump(ROOT/'logs/completion_pipeline_status.json',dict(status='waiting_generation',new_API_calls_by_pipeline=0))
    while len(list((ROOT/'results').glob('*.json')))<500:
        status=read(ROOT/'logs/generation_status.json') if (ROOT/'logs/generation_status.json').exists() else {}
        if status.get('stop_reason'):
            dump(ROOT/'logs/completion_pipeline_status.json',dict(status='generation_paused',reason=status['stop_reason'],new_API_calls_by_pipeline=0));return
        time.sleep(15)
    # The generation process may still be completing its last status write.
    for _ in range(20):
        db=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True)
        n=db.execute("SELECT COUNT(*) FROM calls WHERE state='started'").fetchone()[0];db.close()
        if n==0:break
        time.sleep(2)
    assert n==0
    lock=read(ROOT/'inputs/generation_locked.json')
    assert all(sha(ROOT/name)==v for name,v in lock['files'].items())
    assert all(sha(p)==h for p,h in read(ROOT/'inputs/protected_sources.json').items())
    dump(ROOT/'logs/completion_pipeline_status.json',dict(status='auditing_raw_replies',new_API_calls_by_pipeline=0))
    run([str(ROOT/'code/audit_results.py')])
    # Repeat the same calculations into a separate output directory; only output
    # paths change, and all frozen inputs/raw responses remain identical.
    repeat=ROOT/'analysis_repeat';repeat.mkdir(exist_ok=False)
    text=(ROOT/'code/audit_results.py').read_text().replace("ROOT/'analysis'","ROOT/'analysis_repeat'")
    wrapper="import sys;sys.path.insert(0,"+repr(str(ROOT/'code'))+");exec(compile("+repr(text)+",'audit_repeat_output_only','exec'),{'__name__':'__main__'})"
    run(['-c',wrapper])
    numeric=sorted(p.name for p in (ROOT/'analysis').glob('*.csv') if p.name not in ['reference_distribution_summary.csv','pilot_positions8.csv','pilot_candidates48.csv'])
    assert len(numeric)==8 and all((repeat/name).read_bytes()==(ROOT/'analysis'/name).read_bytes() for name in numeric)
    dump(ROOT/'analysis/fresh_reproduction_verification.json',dict(status='passed',numeric_files_identical=numeric,independent_output_directory=True,new_API_calls=0))
    run([str(ROOT/'code/plot_results.py')])
    db=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True)
    costs=dict(call_states=dict(db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state').fetchall()),
        estimated_usage_cost_cny=db.execute('SELECT COALESCE(SUM(charge_estimate),0) FROM calls').fetchone()[0],
        conservative_peak_cost_cny=db.execute('SELECT COALESCE(SUM(charge_peak),0) FROM calls').fetchone()[0],API_cost_cap_cny=None)
    db.close();dump(ROOT/'analysis/usage_summary.json',costs)
    with (ROOT/'analysis/结果与解读.md').open('a',encoding='utf-8') as f:
        f.write('\n\n按用量估算的本轮API费用：'+f"{costs['estimated_usage_cost_cny']:.2f}"+'元，不含服务器租赁；用户未设置费用上限。原始调用状态：'+json.dumps(costs['call_states'])+'。\n')
        f.write('\n八份主要数值CSV在独立输出目录重新计算逐字节一致。两张图共四个PNG/SVG文件，源数据与坐标在figures/figure_index.json及plotted_coordinates.csv。\n')
    assert all(sha(ROOT/name)==v for name,v in lock['files'].items())
    assert all(sha(p)==h for p,h in read(ROOT/'inputs/protected_sources.json').items())
    dump(ROOT/'logs/completion_pipeline_status.json',dict(status='packaging',new_API_calls_by_pipeline=0))
    delivery=ROOT/'delivery';delivery.mkdir(exist_ok=True)
    core=[]
    for folder in ['analysis','results','code']:
        core.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    for name in ['protocol.json','AUTHORIZATION.json','README_执行方案.md','inputs/references225.csv','inputs/reference_pools25.csv',
        'inputs/tasks25.json','inputs/schedule500.json','inputs/preparation_checks.json','inputs/runner_self_tests.json',
        'inputs/generation_locked.json','inputs/protected_sources.json','logs/generation_status.json','logs/balance_status.json']:
        core.append(ROOT/name)
    allfiles=[p for p in ROOT.rglob('*') if p.is_file() and 'delivery' not in p.relative_to(ROOT).parts and '__pycache__' not in p.parts]
    # Scan the actual secret privately and never serialize it; project provider is not copied.
    import sys
    sys.path.insert(0,str(ROOT/'code'));from service import settings
    base,secret=settings();needles=[secret.encode()]
    import re
    assert all(not re.search(rb'(?m)^-----BEGIN (?:OPENSSH |RSA |EC |ENCRYPTED )?PRIVATE KEY-----\s*$',p.read_bytes()) for p in allfiles)
    assert all(not any(n in p.read_bytes() for n in needles) for p in allfiles)
    index=dict(status='complete',planned_positions=500,original_sources_unchanged=True,no_manuscript_edits=True,no_public_upload=True,
        credentials_not_in_archives=True,estimated_usage_cost_cny=costs['estimated_usage_cost_cny'],API_cost_cap_cny=None)
    for key,name,files in [('review_archive','review_results_ring500.zip',core),('full_evidence_archive','full_evidence_ring500.zip',allfiles)]:
        dest=delivery/name
        with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(set(files)):z.write(p,p.relative_to(ROOT).as_posix())
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        index[key]=dict(path=str(dest.relative_to(ROOT)),bytes=dest.stat().st_size,sha256=sha(dest),files=len(set(files)))
    dump(delivery/'delivery_index.json',index)
    dump(ROOT/'logs/completion_pipeline_status.json',dict(status='complete',new_API_calls_by_pipeline=0))
    print(json.dumps(index),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        dump(ROOT/'logs/completion_pipeline_status.json',dict(status='needs_attention',error_type=type(exc).__name__,error=str(exc),new_API_calls_by_pipeline=0))
        raise
