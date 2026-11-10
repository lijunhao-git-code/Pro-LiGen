"""Durable five-turn generation with pilot gate, bounded retries and balance-safe resume."""
import argparse,fcntl,hashlib,json,os,sqlite3,threading,time
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from common import ROOT,sha,dump,read,read_jsonl,now,candidate_records,choose,feedback
from service import get_balance,settings,usage_estimate


class Pause(Exception):pass
class TechnicalFailure(Exception):pass


class Runner:
    def __init__(self,root=ROOT,client=None,balance_provider=None):
        self.root=Path(root);self.lock=threading.RLock();self.stop=threading.Event();self.reason=None
        self.balance_provider=balance_provider or get_balance;self.balance=None;self.balance_clock=0.;self.balance_spent=0.;self.calls_since_balance=0
        for name in ['logs/slot_states','requests','responses','results']:(self.root/name).mkdir(parents=True,exist_ok=True)
        self.db=sqlite3.connect(self.root/'logs/usage.sqlite',check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, slot TEXT, attempt INTEGER, turn INTEGER, state TEXT, response_path TEXT, request_sha TEXT, submission INTEGER, upper REAL, charge_peak REAL, charge_estimate REAL, usage TEXT, http_status INTEGER)')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT)');self.db.commit()
        self.client=client
        if client is None:
            from openai import OpenAI
            base,key=settings();self.client=OpenAI(api_key=key,base_url=base,timeout=300,max_retries=0)

    def halt(self,reason):
        self.reason=reason;self.stop.set();raise Pause(reason)

    def peak_accounted(self):
        return float(self.db.execute('SELECT COALESCE(SUM(charge_peak),0) FROM calls WHERE state != ?',('started',)).fetchone()[0])

    def balance_guard(self,required):
        # Called while the same mutex also protects inserts/reservations.
        if self.stop.is_set():raise Pause(self.reason or 'stopped')
        refresh=self.balance is None or time.monotonic()-self.balance_clock>=15 or self.calls_since_balance>=8
        if refresh:
            try:current=self.balance_provider()
            except Exception as exc:
                dump(self.root/'logs/balance_status.json',dict(status='paused_balance_check_failed',error_class=type(exc).__name__,time_utc=now()))
                self.halt('balance_check_failed')
            self.balance=current;self.balance_clock=time.monotonic();self.balance_spent=self.peak_accounted();self.calls_since_balance=0
            with (self.root/'logs/balance_snapshots.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(current)+'\n');stream.flush();os.fsync(stream.fileno())
        active=float(self.db.execute("SELECT COALESCE(SUM(upper),0) FROM calls WHERE state='started'").fetchone()[0])
        after=max(0.,self.peak_accounted()-self.balance_spent)
        safe=self.balance['balance_cny']-active-after
        status=dict(status='available',time_utc=now(),last_live_balance_cny=self.balance['balance_cny'],
            reserved_in_flight_cny=active,peak_accounted_since_check_cny=after,safe_estimated_available_cny=safe,reserve_cny=2.)
        if not self.balance['is_available'] or safe<required+2.:
            status['status']='paused_insufficient_balance';dump(self.root/'logs/balance_status.json',status)
            self.halt('insufficient_balance')
        dump(self.root/'logs/balance_status.json',status)

    def complete(self,call_id,body,started,response_path):
        usage=body.get('usage') or {};peak=usage_estimate(usage,started,True);estimate=usage_estimate(usage,started)
        previous=self.db.execute('SELECT upper FROM calls WHERE id=?',(call_id,)).fetchone()
        if peak is None:peak=previous[0]
        if estimate is None:estimate=peak
        self.db.execute('UPDATE calls SET state=?,response_path=?,charge_peak=?,charge_estimate=?,usage=?,http_status=NULL WHERE id=?',
            ('ok',str(response_path.relative_to(self.root)),peak,estimate,json.dumps(usage),call_id));self.db.commit()
        actual=body.get('model')
        if not actual or 'flash' not in str(actual).lower():self.halt('unexpected_response_model')
        pinned=self.db.execute("SELECT value FROM meta WHERE key='response_model'").fetchone()
        if pinned and pinned[0]!=actual:self.halt('response_model_identifier_changed')
        if not pinned:self.db.execute('INSERT INTO meta VALUES (?,?)',('response_model',actual));self.db.commit()

    def call(self,slot,attempt,turn,messages):
        call_id=f'{slot}_a{attempt}_t{turn}'
        request=dict(model='deepseek-flash',messages=messages,stream=False,max_tokens=16384,
            reasoning_effort='low',thinking={'type':'enabled'})
        fingerprint=hashlib.sha256(json.dumps(request,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        with self.lock:
            prior=self.db.execute('SELECT state,response_path,request_sha,submission FROM calls WHERE id=?',(call_id,)).fetchone()
            if prior:
                if prior[2]!=fingerprint:self.halt('cached_request_hash_mismatch')
                if prior[0] in ['ok','started'] and prior[1] and (self.root/prior[1]).exists():
                    stored=read(self.root/prior[1]);body=stored['response']
                    if prior[0]=='started':self.complete(call_id,body,stored['started_utc'],self.root/prior[1])
                    return body
                if prior[0]=='started':self.halt('uncertain_started_call_without_response')
                if prior[0]=='technical_error':raise TechnicalFailure('previous_technical_failure')
                if prior[0]=='fatal_error':self.halt('previous_fatal_configuration_error')
            upper=((len(json.dumps(messages,ensure_ascii=False).encode())+4096)*2.+16384*8.)/1e6
            self.balance_guard(upper)
            submission=(prior[3]+1) if prior else 1
            stem=f'{call_id}_s{submission}';response_path=self.root/'responses'/f'{stem}.json';started=now()
            dump(self.root/'requests'/f'{stem}.json',dict(call_id=call_id,submission=submission,slot_id=slot,attempt=attempt,turn=turn,
                started_utc=started,request_sha256=fingerprint,request=request,reserved_peak_upper_cny=upper,credentials_recorded=False))
            self.db.execute('INSERT OR REPLACE INTO calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                (call_id,slot,attempt,turn,'started',str(response_path.relative_to(self.root)),fingerprint,submission,upper,0.,0.,'{}',None))
            self.db.commit();self.calls_since_balance+=1
        began=time.monotonic()
        try:
            result=self.client.chat.completions.create(model=request['model'],messages=messages,stream=False,
                max_tokens=request['max_tokens'],reasoning_effort='low',extra_body={'thinking':request['thinking']})
            body=result.model_dump(mode='json')
            dump(response_path,dict(call_id=call_id,submission=submission,started_utc=started,finished_utc=now(),
                elapsed_seconds=time.monotonic()-began,response=body))
            with self.lock:self.complete(call_id,body,started,response_path)
            return body
        except Pause:raise
        except Exception as exc:
            status=getattr(exc,'status_code',None)
            failure=dict(call_id=call_id,submission=submission,error_class=type(exc).__name__,http_status=status,
                finished_utc=now(),elapsed_seconds=time.monotonic()-began)
            dump(self.root/'responses'/f'{stem}_error.json',failure)
            with self.lock:
                if status==402:
                    self.db.execute('UPDATE calls SET state=?,charge_peak=0,charge_estimate=0,http_status=? WHERE id=?',('balance_rejected',status,call_id));self.db.commit()
                    dump(self.root/'logs/balance_status.json',dict(status='paused_API_402',time_utc=now(),progress_saved=True))
                    self.halt('API_insufficient_balance_402')
                if status in [400,401,403,404]:
                    self.db.execute('UPDATE calls SET state=?,charge_peak=0,charge_estimate=0,http_status=? WHERE id=?',('fatal_error',status,call_id));self.db.commit()
                    self.halt('fatal_API_configuration_'+str(status))
                self.db.execute('UPDATE calls SET state=?,charge_peak=?,charge_estimate=?,http_status=? WHERE id=?',('technical_error',upper,upper,status,call_id));self.db.commit()
            raise TechnicalFailure(type(exc).__name__) from None

    def slot(self,slot,entry):
        destination=self.root/'results'/f'{slot["slot_id"]}.json'
        if destination.exists():return read(destination)
        state_path=self.root/'logs/slot_states'/f'{slot["slot_id"]}.json'
        state=read(state_path) if state_path.exists() else dict(**slot,next_attempt=1,attempts=[],started_utc=now())
        for attempt in range(state['next_attempt'],3):
            messages=[dict(role='system',content='你是分子设计助手。只依据输入提出设计假设；外部程序负责实际化学校验、性质与对接，不得虚构工具或实验结果。')]
            candidates=[];extra_tags=0
            try:
                for turn,prompt in enumerate(entry['prompt'],1):
                    if turn==5:prompt+=feedback(candidates,entry)
                    messages.append(dict(role='user',content=prompt))
                    body=self.call(slot['slot_id'],attempt,turn,messages)
                    choice=body['choices'][0];content=choice['message'].get('content') or ''
                    if choice.get('finish_reason')!='stop' or not content.strip():raise TechnicalFailure('empty_or_truncated_response')
                    messages.append(dict(role='assistant',content=content))
                    if turn in [4,5]:
                        proposed,extra=candidate_records(content,entry,turn);candidates.extend(proposed);extra_tags+=extra
                    state.update(current_attempt=attempt,completed_turn=turn,next_attempt=attempt,last_update_utc=now(),status='running',candidates=candidates)
                    dump(state_path,state)
                selected=choose(candidates)
                result=dict(**slot,started_utc=state['started_utc'],finished_utc=now(),technical_complete=True,
                    status='eligible_output' if selected else 'no_eligible_candidate',ok=selected is not None,
                    canonical=selected['canonical'] if selected else None,selected_candidate=selected,candidates=candidates,
                    ignored_extra_tags=extra_tags,successful_attempt=attempt,earlier_attempts=state['attempts'])
                assert len(candidates)==6;dump(destination,result)
                state.update(status='complete',last_update_utc=now());dump(state_path,state);return result
            except Pause as exc:
                state.update(status='paused',pause_reason=str(exc),current_attempt=attempt,next_attempt=attempt,
                    last_update_utc=now(),candidates=candidates);dump(state_path,state)
                return dict(slot_id=slot['slot_id'],status='paused',reason=str(exc))
            except TechnicalFailure as exc:
                state['attempts'].append(dict(attempt=attempt,status='technical_failure',reason=str(exc)))
                state.update(next_attempt=attempt+1,last_update_utc=now(),status='retry_pending');dump(state_path,state)
            except Exception as exc:
                with self.lock:self.reason='unexpected_'+type(exc).__name__;self.stop.set()
                state.update(status='paused_unexpected',error_class=type(exc).__name__,current_attempt=attempt,next_attempt=attempt,last_update_utc=now())
                dump(state_path,state)
                return dict(slot_id=slot['slot_id'],status='paused',reason=self.reason)
        result=dict(**slot,status='failed_after_bounded_attempts',ok=False,technical_complete=False,canonical=None,
            selected_candidate=None,candidates=[],earlier_attempts=state['attempts'],finished_utc=now())
        dump(destination,result);return result

    def summary(self,mode):
        paths=list((self.root/'results').glob('*.json'));results=[read(p) for p in paths]
        with self.lock:
            counts=dict(self.db.execute('SELECT state,COUNT(*) FROM calls GROUP BY state').fetchall())
            estimated=float(self.db.execute('SELECT COALESCE(SUM(charge_estimate),0) FROM calls').fetchone()[0])
        value=dict(time_utc=now(),mode=mode,planned=1000,completed=len(results),delivered=sum(r['ok'] for r in results),
            technical_complete=sum(r.get('technical_complete',False) for r in results),stop_reason=self.reason,
            call_states=counts,estimated_usage_cost_cny=estimated,paid_calls_cached_and_durable=True)
        dump(self.root/'logs/generation_status.json',value);return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');p.add_argument('--full',action='store_true');args=p.parse_args()
    if args.pilot==args.full:p.error('Choose exactly one of --pilot or --full')
    frozen=read(ROOT/'inputs/generation_locked.json')
    assert all(sha(ROOT/name)==v for name,v in frozen['files'].items()),'Frozen code or inputs changed'
    prior_status=ROOT/'logs/generation_status.json'
    if prior_status.exists():
        reason=read(prior_status).get('stop_reason')
        assert not reason or reason in ['insufficient_balance','API_insufficient_balance_402','balance_check_failed'], 'Previous configuration/model/uncertain-call halt requires documented reconciliation'
    if args.full:
        review=read(ROOT/'analysis/pilot_review.json');assert review.get('allow_full_run') is True
        assert all(sha(ROOT/name)==value for name,value in review['pilot_output_hashes'].items())
    guard=(ROOT/'logs/generation.lock').open('a');fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    entries={r['slot_id']:r for r in read_jsonl(ROOT/'inputs/entries1000.jsonl')}
    schedule=read(ROOT/'inputs/schedule1000.json');selected=[s for s in schedule if s['pilot']] if args.pilot else schedule
    pending=[s for s in selected if not (ROOT/'results'/f'{s["slot_id"]}.json').exists()]
    runner=Runner();workers=4 if args.pilot else 8
    print('RUN',('pilot' if args.pilot else 'full'),'pending',len(pending),'workers',workers,'resume cache enabled',flush=True)
    queue=iter(pending)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures={}
        def enqueue():
            if runner.stop.is_set():return False
            try:slot=next(queue)
            except StopIteration:return False
            futures[pool.submit(runner.slot,slot,entries[slot['slot_id']])]=slot;return True
        for _ in range(workers):enqueue()
        while futures:
            done,_=wait(futures,return_when=FIRST_COMPLETED,timeout=30)
            if not done:
                print('HEARTBEAT',json.dumps(runner.summary('pilot' if args.pilot else 'full')),flush=True);continue
            for future in done:
                slot=futures.pop(future)
                try:result=future.result();print('SLOT',slot['slot_id'],result['status'],flush=True)
                except Exception as exc:
                    runner.reason='worker_exception_'+type(exc).__name__;runner.stop.set()
                runner.summary('pilot' if args.pilot else 'full');enqueue()
    value=runner.summary('pilot' if args.pilot else 'full')
    print('GENERATION STATUS',json.dumps(value),flush=True)
    if runner.stop.is_set():raise SystemExit(2)


if __name__=='__main__':main()
