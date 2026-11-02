"""Independently recompute saved responses, all6,000 nominal proposals and target tests."""
import argparse,json,re,sqlite3,time
from collections import defaultdict
import numpy as np,pandas as pd
from common import ROOT,PROPERTIES,read,read_jsonl,sha,now
from analysis_utils import describe,structure,csv,save_json,choose_independent,policy_support,clean
from retained_statistics import paired,correct_holm,interval
POLICIES=['first_eligible','random_unique_expected','best_QED']
GROUPS=['T','M','G','R']
METRICS=['S','U','QED','SA_raw','property_value','target_error','abs_target_error','delta_property',
    'tanimoto','MW','HeavyAtoms','RC','delta_MW','delta_HeavyAtoms','delta_RC',
    'library_exact_match','finetuning_train_exact_match','finetuning_val_exact_match']


def independent_candidate(raw,entry,turn,pos):
    d=describe(raw);row=dict(turn=turn,candidate_position=pos,proposal_index=(turn-4)*3+pos,
        smiles=raw,valid=d is not None,property_hit=False,eligible=False,copy_seed=False,copy_exemplar=False)
    if d is None:return dict(row,reason='invalid_smiles' if raw else 'missing_tagged_proposal')
    row.update(d);value=d[entry['property']]
    row.update(property_value=value,target_error=value-entry['target_center'],abs_target_error=abs(value-entry['target_center']),
        delta_property=value-entry['seed_property'],property_hit=abs(value-entry['target_center'])<=entry['tolerance']+1e-8,
        copy_seed=d['canonical']==entry['seed_smiles'],copy_exemplar=d['canonical'] in entry['exemplars'])
    if d['fragments']!=1:reason='multiple_fragments'
    elif not d['contains_carbon']:reason='no_carbon'
    elif row['copy_seed'] or row['copy_exemplar']:reason='exact_input_copy'
    elif not row['property_hit']:reason='outside_property_band'
    else:reason='eligible'
    row.update(eligible=reason=='eligible',reason=reason,**structure(d['canonical'],entry['seed_smiles']))
    return row


def membership_sets():
    library=set(pd.read_csv(ROOT/'source/library_original.csv').canonical)
    train={};val={}
    for prop in PROPERTIES:
        for split,destination in [('train',train),('val',val)]:
            frame=pd.read_csv(ROOT/f'training/{prop}_{split}_triplets.csv')
            destination[prop]=set(frame[['anchor_smiles','positive_smiles','negative_smiles']].to_numpy().ravel())
    return library,train,val


def memberships(smiles,prop,sets):
    library,train,val=sets
    return dict(library_exact_match=smiles in library,finetuning_train_exact_match=smiles in train[prop],
        finetuning_val_exact_match=smiles in val[prop])


def selected_record(entry,rows,sets,policy='best_QED',k=6):
    support=policy_support(rows,policy);selected=support[0] if len(support)==1 and policy!='random_unique_expected' else None
    out={name:entry[name] for name in ['slot_id','property','chemblid','group','position','seed_smiles','seed_property','target_center','tolerance']}
    out.update(policy=policy,k=k,S=int(bool(support)),U=float(np.mean([r['QED'] for r in support])) if support else 0.,
        canonical=selected['canonical'] if selected else None,eligible_positions=sum(r['eligible'] for r in rows),
        eligible_unique=len({r['canonical'] for r in rows if r['eligible']}))
    for metric in METRICS[2:]:
        vals=[r.get(metric) for r in support]
        out[metric]=float(np.mean(vals)) if vals and all(v is not None for v in vals) else None
    return out


def analyze(outdir,require_complete=True):
    started=time.monotonic();outdir.mkdir(parents=True,exist_ok=True)
    frozen=read(ROOT/'inputs/generation_locked.json')
    assert all(sha(ROOT/name)==value for name,value in frozen['files'].items())
    entries=read_jsonl(ROOT/'inputs/entries1000.jsonl');schedule=read(ROOT/'inputs/schedule1000.json')
    entrymap={e['slot_id']:e for e in entries};assert len(entrymap)==1000
    found=[e for e in entries if (ROOT/'results'/f'{e["slot_id"]}.json').exists()]
    if require_complete:assert len(found)==1000,'All planned positions must finish, including failures'
    sets=membership_sets();grid=[];final=[];policies=[];request_checks=0;result_hashes={};extras=0;attempt2=0;technical_failed=0
    for entry in found:
        sid=entry['slot_id'];path=ROOT/'results'/f'{sid}.json';saved=read(path);result_hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        assert abs(describe(entry['seed_smiles'])[entry['property']]-entry['seed_property'])<1e-9
        assert abs(entry['target_center']-(entry['seed_property']+PROPERTIES[entry['property']]['delta']))<1e-9
        rows=[];slot_extras=0
        if saved['technical_complete']:
            attempt=saved['successful_attempt'];attempt2+=int(attempt==2)
            history=[];saved_candidates={(c['turn'],c['candidate_position']):c for c in saved['candidates']}
            assert len(saved_candidates)==6
            for turn in range(1,6):
                matches=sorted((ROOT/'responses').glob(f'{sid}_a{attempt}_t{turn}_s[0-9]*.json'))
                matches=[p for p in matches if not p.stem.endswith('_error')]
                assert len(matches)==1,(sid,turn,len(matches))
                response=read(matches[0]);body=response['response'];choice=body['choices'][0]
                assert choice['finish_reason']=='stop' and choice['message'].get('content','').strip()
                assert body['model']=='deepseek-flash'
                request=read(ROOT/'requests'/matches[0].name)['request']
                assert request['model']=='deepseek-flash' and len(request['messages'])==2*turn
                prompt=request['messages'][-1]['content']
                if turn<5:assert prompt==entry['prompt'][turn-1]
                else:
                    assert prompt.startswith(entry['prompt'][4])
                    marker='【程序实际计算的上一轮结果】\n'
                    feedback_json=prompt.split(marker,1)[1].split('\n当前目标仅为',1)[0]
                    actual_feedback=json.loads(feedback_json);assert len(actual_feedback)==3
                    for f,c in zip(actual_feedback,rows):
                        assert f['eligible']==c['eligible'] and f['valid']==c['valid']
                        if c['valid']:
                            assert f['canonical']==c['canonical']
                            for metric in ['property_value','delta_property','target_error','QED','SA_raw']:assert abs(f[metric]-c[metric])<1e-9
                if turn>1:
                    assert request['messages'][1:-1]==history
                history=request['messages'][1:]+[dict(role='assistant',content=choice['message']['content'])]
                request_checks+=1
                if turn not in [4,5]:continue
                tags=re.findall(r'<smiles>\s*(.*?)\s*</smiles>',choice['message']['content'],flags=re.I|re.S)
                slot_extras+=max(0,len(tags)-3)
                for pos in range(1,4):
                    raw=tags[pos-1].strip() if pos<=len(tags) else None
                    row=independent_candidate(raw,entry,turn,pos);old=saved_candidates[(turn,pos)]
                    row.update(raw_response_path=matches[0].relative_to(ROOT).as_posix(),
                        raw_request_path='requests/'+matches[0].name,raw_response_sha256=sha(matches[0]))
                    for key in ['valid','property_hit','eligible','smiles']:assert row[key]==old.get(key),(sid,turn,pos,key)
                    if row['valid']:
                        assert row['canonical']==old['canonical']
                        for key in ['logp','tpsa','QED','SA_raw','MW','HeavyAtoms','RC','fragments']:assert abs(row[key]-old[key])<1e-9
                    rows.append(row)
            assert slot_extras==saved.get('ignored_extra_tags',0)
            extras+=slot_extras
        else:
            technical_failed+=1
            assert not saved['ok'] and not saved.get('canonical')
            rows=[dict(turn=4+(i-1)//3,candidate_position=1+(i-1)%3,proposal_index=i,valid=False,property_hit=False,
                eligible=False,smiles=None,reason='no_technically_complete_attempt') for i in range(1,7)]
        assert len(rows)==6
        for row in rows:
            row.update({name:entry[name] for name in ['slot_id','property','chemblid','group','position']})
            row['successful_attempt']=saved.get('successful_attempt');row['technical_complete']=saved['technical_complete']
            row['valid_QED_yield']=row['QED'] if row['valid'] else 0.
            row['eligible_QED_yield']=row['QED'] if row['eligible'] else 0.
            if row['valid']:row.update(memberships(row['canonical'],entry['property'],sets))
        selected=choose_independent(rows)
        assert (selected['canonical'] if selected else None)==saved['canonical'] and bool(selected)==saved['ok']
        record=selected_record(entry,rows,sets);record.update(technical_complete=saved['technical_complete'],generation_status=saved['status'],
            successful_attempt=saved.get('successful_attempt'),source_result=path.relative_to(ROOT).as_posix(),source_result_sha256=result_hashes[path.relative_to(ROOT).as_posix()])
        final.append(record);grid.extend(rows)
        for k in range(1,7):
            current=[selected_record(entry,rows[:k],sets,p,k) for p in POLICIES]
            assert len({r['S'] for r in current})==1
            assert current[-1]['U']+1e-12>=max(r['U'] for r in current)
            policies.extend(current)
        for turn in [4,5]:
            r=selected_record(entry,[c for c in rows if c['turn']==turn],sets,'best_QED',3)
            r['policy']=f'round{turn-3}_best_QED';policies.append(r)
    gf=pd.DataFrame(grid);ff=pd.DataFrame(final);pf=pd.DataFrame(policies)
    csv(outdir/'candidate_grid.csv',gf);csv(outdir/'final_positions.csv',ff);csv(outdir/'selection_replay.csv',pf)
    target=ff.groupby(['property','chemblid','group'])[METRICS].mean().reset_index()
    csv(outdir/'target_means.csv',target)
    summary=target.groupby(['property','group'])[METRICS].mean().reset_index()
    counts=ff.groupby(['property','group']).agg(planned=('slot_id','size'),delivered=('S','sum'),technical_complete=('technical_complete','sum'),unique_outputs=('canonical','nunique')).reset_index()
    pooled=ff.groupby(['property','group']).agg(pooled_successful_QED=('QED','mean')).reset_index()
    summary=summary.merge(counts).merge(pooled);csv(outdir/'group_summary.csv',summary)
    ptarget=pf.groupby(['property','chemblid','group','policy','k'])[METRICS].mean().reset_index()
    csv(outdir/'policy_target_means.csv',ptarget)
    psum=ptarget.groupby(['property','group','policy','k'])[METRICS].mean().reset_index();csv(outdir/'policy_summary.csv',psum)
    cq=[]
    for (prop,task,group),part in gf.groupby(['property','chemblid','group']):
        cq.append(dict(property=prop,chemblid=task,group=group,planned=len(part),valid=int(part.valid.sum()),
            property_hit=int(part.property_hit.sum()),eligible=int(part.eligible.sum()),valid_fraction=float(part.valid.mean()),
            property_hit_fraction=float(part.property_hit.mean()),eligible_fraction=float(part.eligible.mean()),
            valid_QED=float(part.loc[part.valid,'QED'].mean()),eligible_QED=float(part.loc[part.eligible,'QED'].mean()),
            valid_QED_yield=float(part.valid_QED_yield.mean()),eligible_QED_yield=float(part.eligible_QED_yield.mean())))
    cq=pd.DataFrame(cq);csv(outdir/'candidate_target_quality.csv',cq)
    csv(outdir/'candidate_quality_summary.csv',cq.groupby(['property','group']).mean(numeric_only=True).reset_index())
    csv(outdir/'candidate_failure_reasons.csv',gf.groupby(['property','group','reason']).size().rename('count').reset_index())
    tests=[]
    if require_complete:
        assert len(gf)==6000 and len(ff)==1000
        assert counts.planned.eq(125).all() and ff.groupby(['property','chemblid','group']).size().eq(5).all()
        for prop in PROPERTIES:
            for family,contrasts in [('primary8',[('M','R'),('G','R')]),('secondary8',[('R','T'),('G','M')])]:
                for outcome in ['S','U']:
                    mat=target[target.property.eq(prop)].pivot(index='chemblid',columns='group',values=outcome)
                    for left,right in contrasts:tests.append(dict(property=prop,family=family,outcome=outcome,
                        comparison=left+'-'+right,**paired((mat[left]-mat[right]).to_numpy())))
        assert len(tests)==16;correct_holm(tests)
    csv(outdir/'declared_tests.csv',tests)
    baselines=[]
    for row in pd.read_csv(ROOT/'inputs/direct_baselines150.csv').to_dict('records'):
        entry=entrymap[f'{row["property"]}_{row["chemblid"]}_1_T'];d=describe(row['canonical'])
        assert d and d['fragments']==1 and d['contains_carbon'] and d['canonical']!=entry['seed_smiles']
        assert abs(d[row['property']]-entry['target_center'])<=entry['tolerance']+1e-8
        for key in ['QED','SA_raw','MW','HeavyAtoms','RC']:assert abs(d[key]-row[key])<1e-9
        baselines.append(dict(row,**structure(row['canonical'],entry['seed_smiles']),S=1,U=d['QED'],
            abs_target_error=abs(row['target_error']),technical_complete=True,
            **memberships(row['canonical'],row['property'],sets)))
    bf=pd.DataFrame(baselines);csv(outdir/'direct_baselines150.csv',bf)
    csv(outdir/'direct_baseline_summary.csv',bf.groupby(['property','group'])[METRICS].mean().reset_index())
    refs=pd.read_csv(ROOT/'inputs/references2250.csv');reference_checks=0;ref_tan_differences=[]
    for row in refs.to_dict('records'):
        entry=entrymap[row['slot_id']];d=describe(row['smiles']);assert d and d['canonical'] in entry['exemplars'] and len(entry['exemplars'])==3
        assert abs(d[row['property']]-entry['target_center'])<=entry['tolerance']+1e-8
        # Prepared fingerprints used float32 division; independently retain float64 values.
        actual_tan=structure(row['smiles'],entry['seed_smiles'])['tanimoto']
        ref_tan_differences.append(abs(actual_tan-row['tanimoto']))
        assert abs(actual_tan-row['tanimoto'])<1e-7
        for key in ['QED','SA_raw','MW','RC']:assert abs(d[key]-row[key])<1e-9
        reference_checks+=1
    refs['tanimoto_prepared_float32']=refs.tanimoto
    refs['tanimoto']=[structure(r['smiles'],entrymap[r['slot_id']]['seed_smiles'])['tanimoto'] for r in refs.to_dict('records')]
    csv(outdir/'references_recomputed2250.csv',refs)
    refmetrics=['property_value','target_error','tanimoto','QED','SA_raw','MW','RC']
    rt=refs.groupby(['property','chemblid','group'])[refmetrics].mean().reset_index()
    csv(outdir/'reference_target_means.csv',rt);csv(outdir/'reference_summary.csv',rt.groupby(['property','group'])[refmetrics].mean().reset_index())
    db=sqlite3.connect('file:'+str(ROOT/'logs/usage.sqlite')+'?mode=ro',uri=True)
    ledger=pd.read_sql_query('SELECT * FROM calls ORDER BY id',db);db.close();csv(outdir/'API_usage_ledger.csv',ledger)
    report=dict(status='passed_complete' if require_complete else 'passed_partial_snapshot',time_utc=now(),completed=len(found),planned=1000,
        candidate_grid_rows=len(grid),technically_failed_positions=technical_failed,second_attempt_completions=attempt2,
        raw_successful_turns_checked=request_checks,reference_records_recomputed=reference_checks,ignored_extra_tags=extras,
        reference_Tanimoto_max_float32_roundoff=max(ref_tan_differences),
        all_raw_SMILES_descriptors_eligibility_feedback_and_selection_match=True,all_frozen_generation_files_match=True,
        primary_tests=8 if require_complete else 0,secondary_tests=8 if require_complete else 0,
        primary_Holm_family='Both properties, M-R and G-R, S/U:8 tests',secondary_Holm_family='Both properties, R-T and G-M, S/U:8 tests',
        conditional_QED_definition='Mean of per-target successful-output means, excluding targets without delivery; pooled successful QED also provided.',
        U_definition='QED for an eligible delivery; zero for any failed planned position. Average five positions first, then25 equally weighted targets.',
        direct_baselines='150 deterministic task/attribute/method outputs, never replicated as independent samples.',
        novelty_scope='Exact stereochemical canonical membership in26,108 reference library and actual current attribute fine-tuning input sets only.',
        estimated_usage_cost_cny=float(ledger.charge_estimate.sum()),new_API_calls=0,elapsed_seconds=round(time.monotonic()-started,3),
        result_hashes=result_hashes,output_numeric_file_hashes={p.name:sha(p) for p in sorted(outdir.glob('*.csv'))})
    save_json(outdir/'property_verification.json',report)
    print(summary.to_string(index=False),flush=True);print(pd.DataFrame(tests).to_string(index=False),flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');p.add_argument('--output');args=p.parse_args()
    out=ROOT/(args.output or ('analysis/partial' if args.partial else 'analysis/properties'))
    report=analyze(out,not args.partial);print('PROPERTY ANALYSIS',report['status'],report['completed'],flush=True)
