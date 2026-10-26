"""Bounded, logged, resumable interleaved T/M/G generation. No quality resampling."""
import argparse
import fcntl
import json
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from common import PROJECT, OUT, dump, now, read_jsonl, verify_locked, candidate_records, select_candidate, optimization_feedback

class Halt(Exception):
    pass

class TechnicalFailure(Exception):
    pass

class Runner:
    def __init__(self, protocol, root=OUT, client=None):
        self.p, self.root = protocol, root
        self.mutex, self.halt = threading.RLock(), threading.Event()
        self.db = sqlite3.connect(root/'logs/usage.sqlite', check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS calls (id TEXT PRIMARY KEY, status TEXT, reserved REAL, charge REAL, payload TEXT)')
        self.db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)')
        self.db.commit()
        if client is None:
            sys.path.insert(0, str(PROJECT))
            import deepseek
            from openai import OpenAI
            assert deepseek._get_base_url().rstrip('/') == self.p['api']['base_url']
            client = OpenAI(api_key=deepseek._get_api_key(), base_url=self.p['api']['base_url'],
                            timeout=self.p['api']['timeout_seconds'], max_retries=0)
        self.client = client

    def spent(self):
        with self.mutex:
            return float(self.db.execute("SELECT COALESCE(SUM(CASE WHEN status='started' THEN reserved ELSE charge END),0) FROM calls").fetchone()[0])

    def usage_cost(self, body, started):
        usage = body.get('usage') or {}
        if not isinstance(usage.get('prompt_tokens'), int) or not isinstance(usage.get('completion_tokens'), int):
            return None
        timestamp = datetime.fromisoformat(started)
        # Saturday/Sunday are off-peak. Weekday public-holiday discounts are not
        # guessed: use peak rate during weekday peak windows (a conservative estimate).
        peak = timestamp.weekday() < 5 and (1 <= timestamp.hour < 4 or 6 <= timestamp.hour < 10)
        rates = self.p['api']['cny_rates_per_million']['peak' if peak else 'off_peak']
        hit = usage.get('prompt_cache_hit_tokens', (usage.get('prompt_tokens_details') or {}).get('cached_tokens', 0))
        hit = max(0, min(int(hit or 0), usage['prompt_tokens']))
        return (hit * rates['cache_hit'] + (usage['prompt_tokens']-hit)*rates['cache_miss']
                + usage['completion_tokens']*rates['output']) / 1e6

    def call(self, slot_id, attempt, turn, messages):
        call_id = f'{slot_id}_a{attempt}_t{turn}'
        response_path = self.root/'responses'/f'{call_id}.json'
        request = dict(model=self.p['api']['model'], messages=messages, stream=False,
                       max_tokens=self.p['api']['max_tokens'], reasoning_effort='low',
                       thinking={'type':'enabled'})
        with self.mutex:
            previous = self.db.execute('SELECT status,payload FROM calls WHERE id=?', (call_id,)).fetchone()
            if previous:
                if previous[0] == 'ok' and response_path.exists():
                    stored = json.loads(response_path.read_text())
                    return stored['response']
                if previous[0] == 'error':
                    raise TechnicalFailure('Previously recorded failed call: ' + call_id)
                self.halt.set()
                raise Halt('Uncertain started call; do not replay or charge twice: ' + call_id)
            if self.halt.is_set():
                raise Halt('Batch stopped; no new calls')
            input_upper = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + 4096
            upper = (input_upper*2. + self.p['api']['max_tokens']*8.) / 1e6
            if self.spent() + upper > self.p['api']['cost_cap_cny']:
                self.halt.set()
                raise Halt('Cost cap would be exceeded by a new request')
            started = now()
            dump(self.root/'requests'/f'{call_id}.json', dict(call_id=call_id, started_utc=started,
                  slot_id=slot_id, attempt=attempt, turn=turn, request=request,
                  reserved_peak_upper_cny=upper, credentials_recorded=False))
            self.db.execute('INSERT INTO calls VALUES (?,?,?,?,?)', (call_id,'started',upper,0.,'{}'))
            self.db.commit()
        t0 = time.monotonic()
        try:
            result = self.client.chat.completions.create(
                model=request['model'], messages=messages, stream=False,
                max_tokens=request['max_tokens'], reasoning_effort=request['reasoning_effort'],
                extra_body={'thinking':request['thinking']})
            body = result.model_dump(mode='json')
            charge = self.usage_cost(body, started)
            usage_unknown = charge is None
            if usage_unknown:
                charge = upper
            dump(response_path, dict(call_id=call_id, started_utc=started, finished_utc=now(),
                 elapsed_seconds=time.monotonic()-t0, response=body,
                 estimated_charge_cny=charge, usage_unknown=usage_unknown))
            with self.mutex:
                self.db.execute('UPDATE calls SET status=?, charge=?, payload=? WHERE id=?',
                                ('ok',charge,json.dumps(body.get('usage') or {}),call_id))
                self.db.commit()
                actual = body.get('model', '')
                pinned = self.db.execute("SELECT value FROM meta WHERE key='actual_model'").fetchone()
                if not actual or 'flash' not in actual.lower():
                    self.halt.set()
                    raise Halt('Unexpected model returned; keep raw response and stop: '+str(actual))
                if pinned and pinned[0] != actual:
                    self.halt.set()
                    raise Halt('Response model changed mid-batch: '+actual)
                if not pinned:
                    self.db.execute('INSERT INTO meta VALUES (?,?)', ('actual_model',actual))
                    self.db.commit()
                if self.spent() > self.p['api']['cost_cap_cny'] or charge > upper + 1e-10:
                    self.halt.set()
                    raise Halt('Cost accounting upper bound violated; stop for review')
            return body
        except Halt:
            raise
        except Exception as exc:
            status = getattr(exc, 'status_code', None)
            # Exception text may contain secrets/headers; record class and HTTP code only.
            failure = dict(error_class=type(exc).__name__, http_status=status,
                           elapsed_seconds=time.monotonic()-t0, finished_utc=now())
            dump(self.root/'responses'/f'{call_id}_error.json', failure)
            with self.mutex:
                # An unanswered or failed request might already have incurred cost.
                self.db.execute('UPDATE calls SET status=?,charge=?,payload=? WHERE id=?',
                                ('error',upper,json.dumps(failure),call_id))
                self.db.commit()
            if status in (400,401,402,403,404):
                self.halt.set()
                raise Halt('Fatal API/configuration status: '+str(status)) from None
            raise TechnicalFailure(type(exc).__name__) from None

    def slot(self, slot, entry):
        path = self.root/'results'/f'{slot["slot_id"]}.json'
        if path.exists():
            return json.loads(path.read_text())
        record = {**slot, 'started_utc':now(), 'attempts':[], 'ok':False, 'smiles':None}
        for attempt in range(1, self.p['api']['max_whole_attempts_per_slot']+1):
            messages = [{'role':'system','content':self.p['system_prompt']}]
            try:
                content = ''
                candidates = []
                ignored = 0
                for turn, prompt in enumerate(entry['prompt'],1):
                    if turn == 5:
                        prompt += optimization_feedback(candidates, entry)
                    messages.append({'role':'user','content':prompt})
                    body = self.call(slot['slot_id'],attempt,turn,messages)
                    choice = body['choices'][0]
                    content = choice['message'].get('content') or ''
                    if choice.get('finish_reason') != 'stop' or not content.strip():
                        raise TechnicalFailure('Empty or non-stop/truncated response at turn '+str(turn))
                    messages.append({'role':'assistant','content':content})
                    if turn in (4,5):
                        proposed, extras = candidate_records(content, turn, entry)
                        candidates.extend(proposed)
                        ignored += extras
                selected = select_candidate(candidates)
                status = 'eligible_output' if selected else 'no_eligible_candidate'
                record.update(ok=selected is not None, smiles=selected['canonical'] if selected else None,
                              status=status, successful_attempt=attempt, finished_utc=now(),
                              candidates=candidates, ignored_extra_tags=ignored, selected_candidate=selected,
                              technical_complete=True)
                record['attempts'].append(dict(attempt=attempt,status=status))
                dump(path,record)
                return record
            except TechnicalFailure as exc:
                record['attempts'].append(dict(attempt=attempt,status='technical_failure',reason=str(exc)))
            except Halt as exc:
                record.update(status='paused_budget_or_attention',reason=str(exc),finished_utc=now())
                dump(self.root/'logs'/f'{slot["slot_id"]}_paused.json', record)
                raise
        record.update(status='failed_after_bounded_attempts',finished_utc=now())
        dump(path,record)
        return record

def tests():
    import tempfile
    from types import SimpleNamespace
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    entry = dict(seed_smiles='c1ccc(-c2ccccc2)cc1', target_rc=1, exemplars=[], prompt=['test']*5)
    rs, extras = candidate_records('<smiles>not_valid</smiles><smiles>CC</smiles><smiles>c1ccccc1</smiles><smiles>C1CCCCC1</smiles>',4,entry)
    assert len(rs)==3 and extras==1 and not rs[0]['valid'] and not rs[1]['eligible']
    assert select_candidate(rs)['canonical']=='c1ccccc1'
    copied = dict(entry,exemplars=['c1ccccc1'])
    assert select_candidate(candidate_records('<smiles>c1ccccc1</smiles>',4,copied)[0]) is None
    calls = []
    response = dict(model='deepseek-flash', choices=[dict(finish_reason='stop',
        message=dict(content='<smiles>c1ccccc1</smiles>'))], usage=dict(prompt_tokens=10,completion_tokens=10))
    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(model_dump=lambda **_:response)
    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    protocol = dict(system_prompt='test', api=dict(model='deepseek-flash',base_url='https://api.deepseek.com',
        max_tokens=16384, cost_cap_cny=10., max_whole_attempts_per_slot=2,
        cny_rates_per_million=dict(peak=dict(cache_hit=.04,cache_miss=2.,output=8.),
                                 off_peak=dict(cache_hit=.02,cache_miss=1.,output=4.))))
    with tempfile.TemporaryDirectory(prefix='gin_downstream_test_') as scratch:
        root=Path(scratch)
        for name in ('logs','requests','responses','results'):
            (root/name).mkdir()
        runner=Runner(protocol,root,fake)
        slot=dict(slot_id='fixture_T_1',chemblid='fixture',group='T',position=1,order=1)
        result=runner.slot(slot,entry)
        assert result['ok'] and result['smiles']=='c1ccccc1' and len(calls)==5
        assert len(result['candidates'])==6 and runner.slot(slot,entry)==result and len(calls)==5
        # No eligible candidates are scientific failures, not additional paid retries.
        badslot=dict(slot,slot_id='fixture_T_2')
        fail=runner.slot(badslot,copied)
        assert not fail['ok'] and fail['technical_complete'] and len(calls)==10
        runner.p['api']['cost_cap_cny']=runner.spent()
        try:
            runner.call('new',1,1,[{'role':'user','content':'test'}])
            raise AssertionError('Budget gate failed')
        except Halt:
            pass
    print('SELF TESTS PASSED: fixed candidate quota, noncopy RC gate, program selection, scientific failure no retry, resume, budget gate',flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pilot',action='store_true')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        tests()
        return
    protocol=verify_locked()
    guard=(OUT/'logs/generation.lock').open('a')
    fcntl.flock(guard.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    runner=Runner(protocol)
    prompts={(r['chemblid'],r['group']):r for r in read_jsonl(OUT/'inputs/prompts.jsonl')}
    schedule=json.loads((OUT/'inputs/schedule.json').read_text())
    selected=schedule[:6] if args.pilot else schedule
    pending=[s for s in selected if not (OUT/'results'/f'{s["slot_id"]}.json').exists()]
    print('RUN', 'TECHNICAL_PILOT' if args.pilot else 'FULL', 'pending',len(pending),
          'cost cap',protocol['api']['cost_cap_cny'],'SDK retries 0',flush=True)
    t0=time.monotonic()
    problems=[]
    with ThreadPoolExecutor(max_workers=min(3 if args.pilot else protocol['api']['workers'],max(1,len(pending)))) as pool:
        futures={pool.submit(runner.slot,s,prompts[(s['chemblid'],s['group'])]):s for s in pending}
        for future in as_completed(futures):
            slot=futures[future]
            try:
                result=future.result()
                print('SLOT',slot['slot_id'],result['status'],'estimated cost CNY',round(runner.spent(),4),flush=True)
            except Halt as exc:
                problems.append(dict(slot_id=slot['slot_id'],reason=str(exc)))
                print('HALT',slot['slot_id'],str(exc),flush=True)
            except Exception as exc:
                runner.halt.set()
                problems.append(dict(slot_id=slot['slot_id'],error_class=type(exc).__name__))
                print('UNEXPECTED ERROR',slot['slot_id'],type(exc).__name__,flush=True)
            done=len(list((OUT/'results').glob('*.json')))
            dump(OUT/'logs/progress.json',dict(time_utc=now(),completed_slots=done,planned_slots=protocol['planned_slots'],
                 estimated_cost_cny=runner.spent(),problems=problems,elapsed_seconds=time.monotonic()-t0))
    all_results=[json.loads(p.read_text()) for p in sorted((OUT/'results').glob('*.json'))]
    pilot_complete=len(all_results)>=6 and all((OUT/'results'/f'{s["slot_id"]}.json').exists() for s in schedule[:6])
    technical_ok=pilot_complete and all(json.loads((OUT/'results'/f'{s["slot_id"]}.json').read_text()).get('technical_complete',False) for s in schedule[:6])
    summary=dict(time_utc=now(),planned_slots=protocol['planned_slots'],completed_slots=len(all_results),
                 valid_outputs=sum(r['ok'] for r in all_results),estimated_cost_cny=runner.spent(),
                 technical_pilot_passed=technical_ok,problems=problems,
                 elapsed_seconds=time.monotonic()-t0)
    dump(OUT/'logs/generation_status.json',summary)
    print('GENERATION STATUS',json.dumps(summary),flush=True)
    if problems or (args.pilot and not technical_ok):
        raise SystemExit(2)

if __name__=='__main__':
    main()
