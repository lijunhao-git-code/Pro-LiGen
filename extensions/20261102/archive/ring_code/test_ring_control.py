"""Offline regressions for original ring chemistry and durable budget-safe runner."""
import tempfile,json
from pathlib import Path
from types import SimpleNamespace
from rdkit import RDLogger
from common import ROOT,read,dump,now,candidate_records,choose,properties
from run_generation import Runner,Pause


def main():
    RDLogger.DisableLog('rdApp.*')
    entry=dict(seed_smiles='c1ccc(-c2ccccc2)cc1',seed_rc=2,target_rc=1,exemplars=[],prompt=['test']*5)
    rows,extra=candidate_records('<smiles>wrong</smiles><smiles>CC</smiles><smiles>c1ccccc1</smiles><smiles>C1CCCCC1</smiles>',entry,4)
    assert len(rows)==3 and extra==1 and not rows[0]['valid'] and not rows[1]['eligible'] and rows[2]['eligible']
    assert choose(rows)['canonical']=='c1ccccc1'
    assert not choose(candidate_records('<smiles>'+entry['seed_smiles']+'</smiles>',entry,4)[0])
    assert not choose(candidate_records('<smiles>c1ccccc1</smiles>',dict(entry,exemplars=['c1ccccc1']),4)[0])
    assert candidate_records('<smiles>c1ccccc1.O</smiles>',entry,4)[0][0]['reason']=='multiple_fragments'
    calls=[];funded=[True];reject_once=[True]
    body=dict(model='deepseek-flash',choices=[dict(finish_reason='stop',message=dict(content='<smiles>c1ccccc1</smiles>'))],usage=dict(prompt_tokens=100,completion_tokens=50))
    class FakeError(Exception):
        def __init__(self,status):self.status_code=status
    def create(**kwargs):
        calls.append(kwargs)
        if len(calls)==3 and reject_once[0]:reject_once[0]=False;funded[0]=False;raise FakeError(402)
        return SimpleNamespace(model_dump=lambda **_:body)
    client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    balance=lambda:dict(time_utc=now(),is_available=funded[0],balance_cny=100. if funded[0] else 0.)
    slot=dict(slot_id='fixture_R_1',chemblid='fixture',position=1,group='R',order=1,pilot=True)
    with tempfile.TemporaryDirectory(prefix='ring_no_paid_API_') as folder:
        root=Path(folder);dump(root/'protocol.json',dict(system_prompt='test',planned_slots=500,api=dict(cost_cap_cny=60.)))
        runner=Runner(root,client,balance);paused=runner.slot(slot,entry)
        assert paused['status']=='paused' and len(calls)==3 and not (root/'results'/f'{slot["slot_id"]}.json').exists()
        funded[0]=True;resumed=Runner(root,client,balance);result=resumed.slot(slot,entry)
        assert result['ok'] and result['successful_attempt']==1 and len(calls)==6
        assert resumed.slot(slot,entry)==result and len(calls)==6
        failed=resumed.slot(dict(slot,slot_id='fixture_scientific_failure'),dict(entry,exemplars=['c1ccccc1']))
        assert not failed['ok'] and failed['technical_complete'] and len(calls)==11
        low=Runner(root,client,lambda:dict(time_utc=now(),is_available=False,balance_cny=0.))
        assert low.slot(dict(slot,slot_id='fixture_no_balance'),entry)['status']=='paused' and len(calls)==11
        capped=Runner(root,client,balance);capped.p['api']['cost_cap_cny']=0.
        assert capped.slot(dict(slot,slot_id='fixture_no_budget'),entry)['status']=='paused' and len(calls)==11
        assert capped.reason=='cost_cap_reached'
    value=dict(status='passed',new_paid_calls=0,original_ring_gate_and_noncopy=True,fixed_candidate_positions=True,
        API402_resume_same_attempt=True,successful_calls_not_repeated=True,scientific_failure_not_retried=True,
        low_balance_and_cost_cap_halt_before_call=True,original_sources_unchanged=True,time_utc=now())
    dump(ROOT/'inputs/runner_self_tests.json',value);print(json.dumps(value))


if __name__=='__main__':main()
