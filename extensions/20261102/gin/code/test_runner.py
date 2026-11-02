"""No-API chemistry, scientific-failure and402-resume integration tests."""
import json,tempfile
from pathlib import Path
from types import SimpleNamespace
from rdkit import RDLogger
from common import ROOT,properties,evaluate_smiles,candidate_records,choose,in_band,dump,now
from run_generation import Runner


class FakeError(Exception):
    def __init__(self,status):self.status_code=status


def main():
    RDLogger.DisableLog('rdApp.*')
    seed=properties('CCOc1ccccc1');candidate=properties('COc1ccccc1')
    entry=dict(property='logp',seed_smiles=seed['canonical'],seed_property=seed['logp'],
        target_center=candidate['logp'],tolerance=.2,target_lower=candidate['logp']-.2,
        target_upper=candidate['logp']+.2,exemplars=[],prompt=['test']*5)
    assert in_band(1.2,1.,.2) and in_band(.8,1.,.2) and not in_band(1.201,1.,.2)
    accepted=evaluate_smiles(candidate['canonical'],entry)
    assert accepted['eligible'] and accepted['RC']==seed['RC'],'No hidden ring reduction gate'
    assert evaluate_smiles(seed['canonical'],entry)['reason']=='exact_input_copy'
    assert evaluate_smiles(candidate['canonical'],dict(entry,exemplars=[candidate['canonical']]))['reason']=='exact_input_copy'
    assert evaluate_smiles('CC.O',entry)['reason']=='multiple_fragments'
    rows,extra=candidate_records('<smiles>not_valid</smiles><smiles>'+candidate['canonical']+'</smiles><smiles>CC</smiles><smiles>CCC</smiles>',entry,4)
    assert len(rows)==3 and extra==1 and not rows[0]['valid'] and choose(rows)['canonical']==candidate['canonical']
    tentry=dict(entry,property='tpsa',seed_property=seed['tpsa'],target_center=candidate['tpsa'],tolerance=5.,
                target_lower=candidate['tpsa']-5,target_upper=candidate['tpsa']+5)
    assert evaluate_smiles(candidate['canonical'],tentry)['eligible']
    body=dict(model='deepseek-flash',choices=[dict(finish_reason='stop',message=dict(content='<smiles>'+candidate['canonical']+'</smiles>'))],
              usage=dict(prompt_tokens=100,completion_tokens=50))
    slot=dict(slot_id='fixture_logp_T_1',property='logp',chemblid='fixture',position=1,group='T',order=1,pilot=True)
    calls=[];funded=[True];reject_once=[True]
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==3 and reject_once[0]:reject_once[0]=False;funded[0]=False;raise FakeError(402)
        return SimpleNamespace(model_dump=lambda **_:body)
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    balance=lambda:dict(time_utc=now(),is_available=funded[0],balance_cny=100. if funded[0] else 0.)
    with tempfile.TemporaryDirectory(prefix='single_property_no_api_test_') as folder:
        root=Path(folder);runner=Runner(root,client,balance)
        paused=runner.slot(slot,entry)
        assert paused['status']=='paused' and not (root/'results'/f'{slot["slot_id"]}.json').exists()
        assert len(calls)==3 and runner.stop.is_set()
        funded[0]=True
        resumed=Runner(root,client,balance);result=resumed.slot(slot,entry)
        assert result['technical_complete'] and result['ok'] and result['successful_attempt']==1
        assert len(calls)==6,'Two successful context calls must be cached, not billed again'
        again=resumed.slot(slot,entry);assert again==result and len(calls)==6
        # A scientific no-eligible outcome consumes five calls, with no quality retry.
        bad=dict(entry,exemplars=[candidate['canonical']]);badslot=dict(slot,slot_id='fixture_no_eligible')
        result=resumed.slot(badslot,bad)
        assert not result['ok'] and result['technical_complete'] and len(calls)==11
        # Before-call balance protection must not call the model or create a final failure.
        low=Runner(root,client,lambda:dict(time_utc=now(),is_available=False,balance_cny=0.))
        low_result=low.slot(dict(slot,slot_id='fixture_low_balance'),entry)
        assert low_result['status']=='paused' and len(calls)==11
    report=dict(status='passed',paid_API_calls=0,chemistry_band_endpoints=True,no_hidden_ring_gate=True,
        copy_and_fragment_rules=True,fixed_three_positions=True,scientific_failure_not_retried=True,
        API402_pause_and_resume_same_attempt=True,successful_context_calls_not_repeated=True,
        low_balance_stops_before_API=True,completed_result_not_repeated=True,time_utc=now())
    dump(ROOT/'inputs/runner_self_tests.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
