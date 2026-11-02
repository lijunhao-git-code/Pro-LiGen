"""Scheduling-only full-run entry; reuses the unchanged frozen Runner for every call/slot.

Reproduce with the original run_generation.py --full at eight workers if desired.
This entry changes concurrency only:8,16,32 after0,16,48 newly completed slots,
and suppresses further increases if any transport/technical errors have occurred.
"""
import fcntl,json
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from common import ROOT,sha,read,read_jsonl,dump,now
from run_generation import Runner


def main():
    frozen=read(ROOT/'inputs/generation_locked.json')
    assert all(sha(ROOT/name)==value for name,value in frozen['files'].items()),'Frozen inputs changed'
    review=read(ROOT/'analysis/pilot_review.json');assert review['allow_full_run'] is True
    assert all(sha(ROOT/name)==value for name,value in review['pilot_output_hashes'].items())
    launch=read(ROOT/'logs/full_launch.json');assert launch['scheduler_sha256']==sha(__file__)
    status_path=ROOT/'logs/generation_status.json'
    if status_path.exists():
        assert read(status_path).get('stop_reason') in [None,'insufficient_balance','API_insufficient_balance_402','balance_check_failed']
    guard=(ROOT/'logs/generation.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries={r['slot_id']:r for r in read_jsonl(ROOT/'inputs/entries1000.jsonl')}
    pending=[r for r in read(ROOT/'inputs/schedule1000.json') if not (ROOT/'results'/f'{r["slot_id"]}.json').exists()]
    queue=iter(pending);runner=Runner();finished=0;limit=8;events=[]
    print('FULL RAMPED pending',len(pending),'initial',limit,'maximum32; frozen per-slot logic unchanged',flush=True)
    with ThreadPoolExecutor(max_workers=32) as pool:
        active={}
        def fill():
            while not runner.stop.is_set() and len(active)<limit:
                try:slot=next(queue)
                except StopIteration:break
                active[pool.submit(runner.slot,slot,entries[slot['slot_id']])]=slot
        fill()
        while active:
            done,_=wait(active,timeout=30,return_when=FIRST_COMPLETED)
            for future in done:
                slot=active.pop(future)
                try:
                    result=future.result();finished+=int(result['status']!='paused')
                    print('SLOT',slot['slot_id'],result['status'],flush=True)
                except Exception as exc:
                    runner.reason='worker_exception_'+type(exc).__name__;runner.stop.set()
            status=runner.summary('full')
            if not done:print('HEARTBEAT',json.dumps(status),flush=True)
            with runner.lock:
                errors=runner.db.execute("SELECT COUNT(*) FROM calls WHERE state IN ('technical_error','fatal_error')").fetchone()[0]
            desired=32 if finished>=48 else (16 if finished>=16 else 8)
            if desired>limit and errors==0 and not runner.stop.is_set():
                limit=desired;events.append(dict(time_utc=now(),new_completed=finished,workers=limit))
                print('CONCURRENCY',limit,flush=True)
            dump(ROOT/'logs/concurrency_status.json',dict(time_utc=now(),workers=limit,active=len(active),new_completed=finished,
                technical_errors=errors,events=events,scientific_inputs_unchanged=True))
            fill()
    status=runner.summary('full');print('GENERATION STATUS',json.dumps(status),flush=True)
    if runner.stop.is_set():raise SystemExit(2)
    assert status['completed']==1000


if __name__=='__main__':main()
