"""Official API credential access and read-only probes; never serialize credentials."""
from datetime import datetime,timezone,timedelta
import sys
import requests
from common import PROJECT,ROOT,now,dump


def settings():
    sys.path.insert(0,str(PROJECT))
    import deepseek
    base=deepseek._get_base_url().rstrip('/')
    if base not in ['https://api.deepseek.com','https://api.deepseek.com/v1']:
        raise RuntimeError('Unexpected API origin; refuse third-party endpoint')
    return base,deepseek._get_api_key()


def get_balance():
    base,key=settings()
    response=requests.get('https://api.deepseek.com/user/balance',headers={'Authorization':'Bearer '+key},timeout=20)
    response.raise_for_status();body=response.json()
    amounts=[r for r in body.get('balance_infos',[]) if r.get('currency')=='CNY']
    if len(amounts)!=1:raise RuntimeError('Expected one CNY balance record')
    return dict(time_utc=now(),is_available=bool(body.get('is_available')),balance_cny=float(amounts[0]['total_balance']))


def get_models():
    base,key=settings()
    response=requests.get(base+'/models',headers={'Authorization':'Bearer '+key},timeout=30)
    response.raise_for_status();return response.json()


def probe():
    balance=get_balance();models=get_models()
    assert any(x.get('id')=='deepseek-flash' for x in models.get('data',[]))
    snapshot=dict(time_utc=now(),balance=balance,model_catalog=models,credentials_recorded=False,
        paid_generation_calls=0,official_origin=True)
    dump(ROOT/'inputs/api_probe.json',snapshot)
    print(__import__('json').dumps(dict(balance=balance,model_ids=[r.get('id') for r in models.get('data',[])],paid_generation_calls=0)),flush=True)


def usage_estimate(usage,started,peak_only=False):
    if not isinstance(usage.get('prompt_tokens'),int) or not isinstance(usage.get('completion_tokens'),int):return None
    dt=datetime.fromisoformat(started).astimezone(timezone(timedelta(hours=8)))
    # Conservatively treat weekday holidays as ordinary peak weekdays.
    peak=peak_only or (dt.weekday()<5 and (9<=dt.hour<12 or 14<=dt.hour<18))
    rates=dict(hit=.04 if peak else .02,miss=2. if peak else 1.,output=8. if peak else 4.)
    cache=usage.get('prompt_cache_hit_tokens',(usage.get('prompt_tokens_details') or {}).get('cached_tokens',0))
    cache=max(0,min(int(cache or 0),usage['prompt_tokens']))
    return (cache*rates['hit']+(usage['prompt_tokens']-cache)*rates['miss']+usage['completion_tokens']*rates['output'])/1e6


if __name__=='__main__':probe()
