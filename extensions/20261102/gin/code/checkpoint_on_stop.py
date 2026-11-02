"""Archive a quiescent generation state after balance pause, error or completion."""
import fcntl,hashlib,json,os,sqlite3,time,zipfile
from pathlib import Path
from common import ROOT,read,sha,now,dump


def main():
    guard=(ROOT/'logs/checkpoint.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        status=read(ROOT/'logs/generation_status.json')
        db=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True)
        active=db.execute("SELECT COUNT(*) FROM calls WHERE state='started'").fetchone()[0]
        # A live process still owns this flock while draining in-flight calls.
        generation_guard=(ROOT/'logs/generation.lock').open('a')
        try:fcntl.flock(generation_guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB);quiescent=True
        except BlockingIOError:quiescent=False
        should_checkpoint=status['completed']==1000 or status.get('stop_reason')
        if should_checkpoint and quiescent:
            dest=ROOT/'archives';dest.mkdir(exist_ok=True)
            tag=now().replace(':','').replace('+','_');stem=f'generation_{status["completed"]}_{tag}'
            snapshot=dest/(stem+'_usage.sqlite')
            backup=sqlite3.connect(snapshot);db.backup(backup);backup.close();db.close()
            files=set()
            frozen=read(ROOT/'inputs/generation_locked.json')
            for name in frozen['files']:files.add(ROOT/name)
            for folder in ['code','requests','responses','results','logs/slot_states']:
                files.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ['.py','.json'])
            for name in ['inputs/generation_locked.json','analysis/pilot_review.json','logs/generation_status.json',
                'logs/balance_status.json','logs/concurrency_status.json','logs/full_launch.json']:
                if (ROOT/name).is_file():files.add(ROOT/name)
            manifest={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
            manifest['logs/usage.sqlite']=sha(snapshot)
            archive=dest/(stem+'.zip')
            with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
                for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
                z.write(snapshot,'logs/usage.sqlite')
                z.writestr('CHECKPOINT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,indent=2))
                z.writestr('CHECKPOINT_STATE.json',json.dumps(dict(time_utc=now(),generation_status=status,uncertain_started_calls=active,
                    safe_to_resume=active==0,notes='Successful replies cached. If uncertain_started_calls>0, reconcile with response files/provider before any resend.'),indent=2))
            with zipfile.ZipFile(archive) as z:assert z.testzip() is None
            receipt=dict(time_utc=now(),archive=str(archive.relative_to(ROOT)),sha256=sha(archive),bytes=archive.stat().st_size,
                completed=status['completed'],stop_reason=status.get('stop_reason'),uncertain_started_calls=active,
                cached_successful_calls_not_repeated=True)
            dump(ROOT/'logs/checkpoint_receipt.json',receipt);print(json.dumps(receipt),flush=True)
            generation_guard.close();return
        db.close();generation_guard.close();time.sleep(20)


if __name__=='__main__':main()
