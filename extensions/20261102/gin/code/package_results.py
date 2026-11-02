"""Create review and full-evidence archives without credentials or mutable SQLite WAL."""
import hashlib,json,re,sqlite3,zipfile
from pathlib import Path
from common import ROOT,read,sha,dump,now


def archive(path,files,extra=None):
    from service import settings
    _,secret=settings()
    private_key_block=re.compile(rb'(?m)^-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----\r?\n[A-Za-z0-9+/=\r\n]{32,}')
    assert len(secret)>=16
    for p in files:
        assert p.name not in ['.env','id_rsa','id_ed25519','autodl_codex']
        content=p.read_bytes()
        assert secret.encode() not in content,'Actual API credential detected; refuse archive: '+p.relative_to(ROOT).as_posix()
        assert not private_key_block.search(content),'Actual private key block detected; refuse archive: '+p.relative_to(ROOT).as_posix()
    manifest={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
        if extra:
            for name,p in extra.items():z.write(p,name);manifest[name]=sha(p)
        z.writestr('ARCHIVE_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert all(hashlib.sha256(z.read(name)).hexdigest()==value for name,value in manifest.items())
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,sha256=sha(path),files=len(manifest))


def main():
    assert read(ROOT/'analysis/fresh_reproduction_verification.json')['status']=='passed'
    assert read(ROOT/'analysis/independent_count_verification.json')['status']=='passed'
    dest=ROOT/'delivery';dest.mkdir(exist_ok=True)
    review=set()
    for p in (ROOT/'analysis').rglob('*'):
        if not p.is_file() or any(part in ['partial','paused_snapshot','mcs_cache'] for part in p.parts):continue
        if p.suffix in ['.csv','.json','.png','.svg','.md']:review.add(p)
    for name in ['protocol.json','README_运行与数据指引.md','inputs/generation_locked.json','inputs/tasks25.json',
        'inputs/reference_pools50.csv','inputs/references2250.csv','inputs/direct_baselines150.csv','inputs/schedule1000.json',
        'training/logp_status.json','training/tpsa_status.json','training/logp_epochs.csv','training/tpsa_epochs.csv',
        'logs/generation_status.json','logs/balance_status.json','logs/post_generation_balance.json','logs/full_launch.json',
        'logs/docking_progress.json','logs/mcs_progress.json','logs/checkpoint_receipt.json','logs/offline_pipeline_status.json','logs/postrun_validation_status.json']:
        if (ROOT/name).exists():review.add(ROOT/name)
    review.update((ROOT/'code').glob('*.py'))
    review.update((ROOT/'results').glob('*.json'))
    if (ROOT/'analysis/manual_sample_slots.json').exists():
        for sid in read(ROOT/'analysis/manual_sample_slots.json')['slots']:
            for folder in ['requests','responses']:review.update((ROOT/folder).glob(sid+'_*.json'))
    review_receipt=archive(dest/'review_results_1000.zip',review)
    full=set(review)
    for name in ['inputs','source','models','training','requests','responses','results','docking']:
        full.update(p for p in (ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith('.tmp'))
    full.update((ROOT/'analysis/mcs_cache').glob('*.json'))
    full.update(p for p in (ROOT/'logs').rglob('*') if p.is_file() and p.suffix in ['.json','.jsonl','.log','.py'])
    # Copy the committed SQLite state through its backup API, never copy an active WAL.
    backup=dest/'usage_committed.sqlite';db=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True);other=sqlite3.connect(backup);db.backup(other);other.close();db.close()
    full_receipt=archive(dest/'full_evidence_1000.zip',full,{'logs/usage.sqlite':backup})
    report=dict(status='complete',time_utc=now(),review_archive=review_receipt,full_evidence_archive=full_receipt,
        no_API_credentials_or_SSH_password_files_included=True,actual_API_key_and_private_key_marker_scan_passed=True,
        private_original_provider_not_copied=True,
        expected_inputs_are_experimental_prompts_and_molecular_data=True)
    dump(dest/'delivery_index.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
