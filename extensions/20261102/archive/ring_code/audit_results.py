"""Rebuild ring candidates from raw replies, then open the fixed pilot gate or analyze."""
import argparse,hashlib,json,math,sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import rdFingerprintGenerator
from common import ROOT,read,read_jsonl,sha,dump,now,candidate_records,choose,feedback,properties
from retained_statistics import paired,correct_holm


def audit(pilot=False):
    RDLogger.DisableLog('rdApp.*')
    schedule=read(ROOT/'inputs/schedule500.json');entries={e['slot_id']:e for e in read_jsonl(ROOT/'inputs/entries500.jsonl')}
    expected=[s for s in schedule if s['pilot']] if pilot else schedule
    rows=[];candidates=[];hashes={};models=set();response_count=0
    for slot in expected:
        sid=slot['slot_id'];path=ROOT/'results'/f'{sid}.json';r=read(path);entry=entries[sid]
        assert all(r[k]==slot[k] for k in ['slot_id','chemblid','group','position'])
        hashes[path.relative_to(ROOT).as_posix()]=sha(path)
        if r.get('technical_complete'):
            attempt=r['successful_attempt'];rebuilt=[]
            messages=[dict(role='system',content=read(ROOT/'protocol.json')['system_prompt'])]
            for turn in range(1,6):
                prefix=f'{sid}_a{attempt}_t{turn}_s'
                paths=[p for p in (ROOT/'responses').glob(prefix+'*.json') if not p.name.endswith('_error.json')]
                assert len(paths)==1,(sid,turn,len(paths))
                rp=paths[0];resp=read(rp);reqpath=ROOT/'requests'/rp.name;req=read(reqpath)
                prompt=entry['prompt'][turn-1]+(feedback(rebuilt,entry) if turn==5 else '')
                messages.append(dict(role='user',content=prompt));assert req['request']['messages']==messages
                fingerprint=hashlib.sha256(json.dumps(req['request'],sort_keys=True,ensure_ascii=False).encode()).hexdigest()
                assert req['request_sha256']==fingerprint
                body=resp['response'];models.add(body['model']);choice=body['choices'][0]
                assert choice['finish_reason']=='stop';content=choice['message']['content'];assert content.strip()
                messages.append(dict(role='assistant',content=content));response_count+=1
                if turn in [4,5]:
                    extra,ignored=candidate_records(content,entry,turn);rebuilt.extend(extra)
                    for c in extra:
                        candidates.append(dict(slot_id=sid,chemblid=slot['chemblid'],group=slot['group'],position=slot['position'],
                            raw_response_path=str(rp),raw_response_sha256=sha(rp),raw_request_path=str(reqpath),**c))
            assert len(rebuilt)==6 and rebuilt==r['candidates'],sid
            selected=choose(rebuilt);assert selected==r['selected_candidate'] and bool(selected)==r['ok']
            if selected:assert selected['canonical']==r['canonical']
        elif pilot:raise AssertionError('Pilot technical failure requires inspection, not automatic bulk launch')
        else:
            assert not r['ok'];rebuilt=[];selected=None
            for turn in [4,5]:
                for pos in range(1,4):candidates.append(dict(slot_id=sid,chemblid=slot['chemblid'],group=slot['group'],position=slot['position'],
                    turn=turn,candidate_position=pos,valid=False,eligible=False,reason='final_technical_failure',smiles=None))
        row=dict(**slot,technical_complete=r.get('technical_complete',False),delivery=int(r['ok']),S=int(r['ok']),
            U=selected['QED'] if selected else 0.,source_result=str(path),source_result_sha256=sha(path),
            canonical=selected['canonical'] if selected else None)
        if selected:
            for k in ['QED','SA_raw','MW','HeavyAtoms','RC']:row[k]=selected[k]
        rows.append(row)
    assert len(models)==1 and next(iter(models)).lower().find('flash')>=0
    report=dict(status='passed',pilot=pilot,positions=len(rows),raw_successful_responses_verified=response_count,
        all_prompts_feedback_candidates_and_final_choices_verified=True,response_models=sorted(models),new_API_calls=0,time_utc=now())
    if pilot:
        assert len(rows)==8
        report.update(allow_full_run=True,pilot_output_hashes=hashes,
            delivered=sum(r['delivery'] for r in rows),gate_is_technical_not_best_success_rate=True)
        dump(ROOT/'analysis/pilot_review.json',report)
        pd.DataFrame(rows).to_csv(ROOT/'analysis/pilot_positions8.csv',index=False)
        pd.DataFrame(candidates).to_csv(ROOT/'analysis/pilot_candidates48.csv',index=False)
        print(json.dumps(report));return
    assert len(rows)==500 and len(candidates)==3000
    frame=pd.DataFrame(rows);cand=pd.DataFrame(candidates)
    lib=pd.read_csv(ROOT/'source/library.csv');keys=set(lib.canonical)
    fpgen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    seed_by_task={t['chemblid']:t['seed_smiles'] for t in read(ROOT/'inputs/tasks25.json')}
    for i,row in frame.iterrows():
        if row.delivery:
            molecule=Chem.MolFromSmiles(row.canonical);seed=Chem.MolFromSmiles(seed_by_task[row.chemblid])
            frame.loc[i,'Tanimoto']=DataStructs.TanimotoSimilarity(fpgen.GetFingerprint(molecule),fpgen.GetFingerprint(seed))
            frame.loc[i,'novel_library_exact']=row.canonical not in keys
            frame.loc[i,'delta_QED']=row.QED-properties(seed_by_task[row.chemblid])['QED']
    target=frame.groupby(['chemblid','group'])[['S','U','QED','SA_raw','MW','Tanimoto','delta_QED']].mean().reset_index()
    summary=[]
    for group in ['T','M','G','R']:
        f=frame[frame.group==group];t=target[target.group==group]
        summary.append(dict(group=group,planned=len(f),delivered=int(f.delivery.sum()),S=f.S.mean(),U=f.U.mean(),
            QED_target_macro=t.QED.mean(),QED_pooled_success=f.QED.mean(),SA_raw=t.SA_raw.mean(),Tanimoto=t.Tanimoto.mean(),
            unique_canonical=f.canonical.nunique(),novel_library_exact=f.novel_library_exact.mean()))
    tests=[]
    for family,contrasts in [('primary4',['M-R','G-R']),('secondary4',['R-T','G-M'])]:
        for contrast in contrasts:
            a,b=contrast.split('-')
            for metric in ['S','U']:
                wide=target.pivot(index='chemblid',columns='group',values=metric)
                tests.append(dict(family=family,comparison=contrast,metric=metric,**paired((wide[a]-wide[b]).to_numpy())))
    correct_holm(tests)
    quality=[];policies=[];curves=[]
    for group in ['T','M','G','R']:
        c=cand[cand.group==group]
        tq=c.groupby('chemblid').apply(lambda x:pd.Series(dict(valid_fraction=x.valid.mean(),eligible_fraction=x.eligible.mean(),
            valid_QED=x.loc[x.valid,'QED'].mean(),eligible_QED=x.loc[x.eligible,'QED'].mean(),
            candidate_QED_contribution=x.QED.fillna(0).sum()/30)),include_groups=False)
        quality.append(dict(group=group,planned_candidates=750,valid=int(c.valid.sum()),eligible=int(c.eligible.sum()),**tq.mean().to_dict()))
    for sid,e in entries.items():
        c=cand[cand.slot_id==sid].sort_values(['turn','candidate_position'])
        for k in range(1,7):
            available=c.iloc[:k];eligible=available[available.eligible].drop_duplicates('canonical',keep='last')
            values={'first_eligible':available[available.eligible].iloc[0].QED if len(eligible) else 0.,
                'uniform_unique_expected':eligible.QED.mean() if len(eligible) else 0.,
                'max_QED':eligible.QED.max() if len(eligible) else 0.}
            for policy,value in values.items():
                curves.append(dict(slot_id=sid,chemblid=e['chemblid'],group=e['group'],position=e['position'],k=k,policy=policy,S=int(len(eligible)>0),U=float(value)))
    replay=pd.DataFrame(curves)
    policy_summary=replay.groupby(['group','policy','k'])[['S','U']].mean().reset_index()
    for name,data in [('final_positions500.csv',frame),('candidate_grid3000.csv',cand),('target_means100.csv',target),
        ('group_summary4.csv',pd.DataFrame(summary)),('declared_tests8.csv',pd.DataFrame(tests)),('candidate_quality4.csv',pd.DataFrame(quality)),
        ('selection_replay9000.csv',replay),('policy_summary72.csv',policy_summary)]:data.to_csv(ROOT/'analysis'/name,index=False)
    report.update(delivered=int(frame.delivery.sum()),unique_canonical=frame.canonical.nunique(),candidate_positions=len(cand),
        all_original_sources_unchanged=all(sha(p)==h for p,h in read(ROOT/'inputs/protected_sources.json').items()))
    dump(ROOT/'analysis/generation_analysis_verification.json',report)
    write_report(summary,tests,report)
    print(json.dumps(report));print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(tests).to_string(index=False))


def write_report(summary,tests,verification):
    lines=['# 减环任务同期随机范例对照：结果与解释','',
        '本轮25任务×5次×T/M/G/R，共500个独立运行位置。只改变输入范例的选择方式；四组同期运行，原减环375位置单独保留。',
        '','T无范例；M/G沿用原Morgan/GIN前三参考；R从同一精确环数池均匀抽取三个，固定随机种子，五次生成和两轮提案内保持不变。没有新训练。',
        '', '| 方法 | 合格/125 | S | U | 条件QED（任务宏平均） | SA | Tanimoto |','|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:lines.append(f"| {r['group']} | {r['delivered']}/125 | {r['S']:.4f} | {r['U']:.4f} | {r['QED_target_macro']:.4f} | {r['SA_raw']:.4f} | {r['Tanimoto']:.4f} |")
    lines+=['','S看计划运行能否交回合格分子；U为未交回记0的QED产出。条件QED/SA/Tanimoto先在任务的合格分子内平均，再对任务平均，不能用不同分母混算。',
        '','| 检验族 | 比较 | 指标 | 差值 | 未校正95%CI | Holm p |','|---|---|---|---:|---|---:|']
    for r in tests:lines.append(f"| {r['family']} | {r['comparison']} | {r['metric']} | {r['difference']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] | {r['p_holm']:.5f} |")
    lines+=['','主要检验为M−R、G−R的S/U共四项，统一Holm；次要R−T、G−M四项单独校正。以25配对任务为单位，不能将重复生成视为独立任务。未显著不代表等效。',
        '','参考环数在每任务内一致，Tanimoto分布不要求相同；排序方法本就选择更接近种子的范例。完整参考统计见reference_distribution_summary.csv和inputs/references225.csv。',
        '','候选整体、首个/随机/最高QED和前1–6位置敏感性另存；这些为固定生成轨迹的选择分析。',
        '','数据入口：final_positions500.csv定位原结果JSON；candidate_grid3000.csv定位原请求、响应及SHA256；declared_tests8.csv给出所有预定检验；generation_analysis_verification.json记录从原回复重建候选和最终选择的核验。',
        '','本轮没有修改论文，未增加对接。结果不与10月11日旧375位置合并为同一同期比较。']
    (ROOT/'analysis/结果与解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');args=parser.parse_args();audit(args.pilot)
