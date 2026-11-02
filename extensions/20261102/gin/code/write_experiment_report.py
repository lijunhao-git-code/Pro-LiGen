"""Write the concise Chinese interpretation directly from verified exported data."""
import json
import pandas as pd
from common import ROOT,read,now,dump


def main():
    fresh=read(ROOT/'analysis/fresh_reproduction_verification.json');assert fresh['status']=='passed'
    pv=read(ROOT/'analysis/properties/property_verification.json');assert pv['status']=='passed_complete'
    sv=read(ROOT/'analysis/docking/docking_source_verification.json');assert sv['status']=='passed'
    p=ROOT/'analysis/properties';df=pd.read_csv(p/'group_summary.csv');tests=pd.read_csv(p/'declared_tests.csv')
    ds=pd.read_csv(ROOT/'analysis/docking/docking_summary.csv');ss=pd.read_csv(ROOT/'analysis/structures/structural_summary.csv')
    policy=pd.read_csv(p/'policy_summary.csv');ff=pd.read_csv(p/'final_positions.csv')
    headline=read(ROOT/'analysis/headline_results.json');balance=read(ROOT/'logs/post_generation_balance.json')
    names={'T':'无范例','M':'Morgan范例','G':'GIN范例','R':'随机达标范例'}
    propnames={'logp':'LogP降低1.0±0.2','tpsa':'TPSA降低20±5 Å²'}
    lines=['# 两项单属性分子生成实验：结果与解读','',
        '本轮完成两个新的单属性任务，并加入随机达标范例对照。结果保存在独立实验目录，尚未写入论文。','',
        '## 1. 实验想回答什么','',
        '第一，原流程从减环换成降低LogP或TPSA以后，是否仍能交回满足条件、保持一定种子结构的分子。第二，如果给模型三个达标范例，按结构或GIN排序挑选范例，是否比从同一候选池随机挑三个更有帮助。第三，GIN相较Morgan是否带来额外收益。','',
        'T不提供范例；M、G、R分别用Morgan、对应属性GIN、均匀随机选择三个范例。M/G/R对每个任务使用同一属性达标池；各方法共用相同种子、文献、靶点信息和生成步骤。','',
        '## 2. 实际怎么做','',
        '每项属性25个任务，每任务每方法独立运行5次，共500个位置；两项合计1000个位置。每个位置独立完成五轮对话，先建立规则并分析文献与靶点，第4轮提出三个候选，收到程序实际计算的反馈后，第5轮再提出三个。从两轮合格候选中选择QED最高者；同分按SA、canonical顺序处理。','',
        '每项实验只限定一个属性。共同基础检查为有效、单连通、含碳、不精确复制种子或本次所给范例。没有额外减环、减重或结构相似度接收门槛。LogP和TPSA均由同一RDKit版本实算。','',
        'GIN分别完成了对应属性的微调，均从同一已有初始化开始、由验证损失选择权重。LogP使用17334个训练三元组和3650个验证三元组，TPSA使用16187和3426个；生成结果没有参与选模型。','',
        '先完成16个预先固定的位置，检查原始回复、实际性质、非复制规则和最终选择后，纳入1000并继续其余984个。并发按8→16→32提高；科学设置始终固定。','',
        '## 3. 交回结果','',
        f'1000个位置全部结束，交回{headline["delivered"]}条合格记录，其中有{headline["unique_canonical_outputs"]}个不同的canonical结构。6000个正式候选位置中，{headline["valid_candidates"]}个可解析，{headline["eligible_candidates"]}个满足完整接收条件。失败位置保留，没有追加生成补成功数。','',
        '| 条件 | 方法 | 合格/计划 | 合格率S | 交回分子平均QED | QED yield U |','| --- | --- | ---: | ---: | ---: | ---: |']
    for prop in ['logp','tpsa']:
        for group in ['T','M','G','R']:
            r=df[df.property.eq(prop)&df.group.eq(group)].iloc[0]
            assert abs(r.U-r.S*r.pooled_successful_QED)<1e-12
            lines.append(f'| {propnames[prop]} | {group}：{names[group]} | {int(r.delivered)}/125 | {100*r.S:.1f}% | {r.pooled_successful_QED:.4f} | {r.U:.4f} |')
    lines.extend(['',
        'S回答“多少计划位置交回了合格结果”。QED列只看交回分子，为本表采用的合并均值；U把未交回位置记0，覆盖全部计划位置。比较方法时优先一起看S与U，不能只看成功结果的QED。统计检验仍以25个任务为单位，每任务五次先平均。原始CSV另外提供按任务等权的条件QED均值，两种条件均值的分母不同。','',
        '这组数据支持流程可以扩展到本次两种数值条件。它没有显示GIN的交回率优势：LogP的M均值最高，TPSA四组均接近全部交回。','',
        '## 4. 范例排序与GIN的作用','',
        '主要比较为M−R、G−R。它们在相同达标池基础上检验“怎样挑范例”是否有额外收益。两属性、两比较、S/U共8项检验统一进行Holm校正。下面列出U的四项；完整S/U结果在declared_tests.csv。','',
        '| 条件 | 比较 | U差值 | Holm校正后p |','| --- | --- | ---: | ---: |'])
    for prop in ['logp','tpsa']:
        for comparison in ['M-R','G-R']:
            r=tests[tests.property.eq(prop)&tests.family.eq('primary8')&tests.outcome.eq('U')&tests.comparison.eq(comparison)].iloc[0]
            lines.append(f'| {propnames[prop]} | {comparison} | {r.difference:+.4f} | {r.p_holm:.4f} |')
    assert tests[tests.family.eq('primary8')].p_holm.ge(.05).all()
    lines.extend(['',
        'M和G相对R的U均值方向为正，但预先确定的8项主要检验中没有通过校正后的显著差异。因此，本轮可以报告改善趋势，不能把排序优势写成已经证实。','',
        'G−M以及R−T属于另一个预先确定的8项次要检验族，本轮也没有检出校正后的显著差异。GIN没有显示出优于Morgan的额外性能证据；随机达标范例相对无范例的提升也很小。适合保留的表述是：GIN作为可选检索实现进行了完整对照，整体流程能够处理本次两种新属性条件。','',
        '这轮没有获得“GIN是性能主要来源”的预期证据。新增的价值主要在任务范围扩展、随机范例对照，以及更清楚地区分流程、选例和最终选择的作用。','',
        '## 5. 最终选择和结构保持说明了什么','',
        '首个合格、随机不同合格结构的精确期望、最高QED三种规则都使用同一批保存候选，没有重新调用模型。各规则的交回与否相同，QED选择会改变交回分子的质量和结构分布。'])
    for prop in ['logp','tpsa']:
        sub=policy[policy.property.eq(prop)&policy.k.eq(6)].pivot(index='group',columns='policy',values='U')
        gain=sub.best_QED-sub.random_unique_expected
        lines.append(f'\n- {propnames[prop]}：最高QED相对随机合格期望的U增加范围为{gain.min():.4f}至{gain.max():.4f}（四组）。')
    gen=ss[ss.group.isin(['T','M','G','R'])];direct=ss[ss.group.str.startswith('LIB_')]
    lines.extend(['',
        '原始候选层面也已单独汇总，避免把最终择优后的QED直接当成整个候选池的质量。保存候选数量曲线展示累计交回和选择变化；反馈的独立贡献继续依据原研究C的匹配对照判断。','',
        f'生成组的各条件/方法平均种子Tanimoto为{gen.tanimoto.min():.3f}至{gen.tanimoto.max():.3f}；直接检索基线为{direct.tanimoto.min():.3f}至{direct.tanimoto.max():.3f}。直接检索的QED通常更高，但种子结构相似度较低。当前流程的价值更适合围绕“满足属性条件的局部编辑及其取舍”组织。','',
        '结构阈值敏感性使用已保存的合格候选重选，完整曲线已导出。MCS有81/2339个分子对搜索达到时限，保留上下界；汇总表将完成搜索的条件均值和全部交回结果的上下界均值明确分列。','',
        '按完整canonical SMILES精确匹配，970条合格输出均未命中本次原始26108条参考库记录。693是不同结构数；其余为不同位置的重复输出。','',
        '## 6. 统一对接','',
        '对接使用预先固定的11个核验靶点和同一套受体、网格、配体准备与Vina设置。全部组的交回结果均按统一规则进入；相同靶点和分子的确定性结果缓存，但每个观察位置仍保留。下表只列生成组，计划分母为每组11×5=55。','',
        '| 条件 | 方法 | 合格交回/55 | 对接完成/55 | 平均ΔVina |','| --- | --- | ---: | ---: | ---: |'])
    for prop in ['logp','tpsa']:
        for group in ['T','M','G','R']:
            r=ds[ds.property.eq(prop)&ds.group.eq(group)&ds.kind.eq('generated')].iloc[0]
            lines.append(f'| {propnames[prop]} | {group} | {int(r.eligible_deliveries)}/55 | {int(r.docking_completed)}/55 | {r.delta_vina:+.3f} |')
    gd=ds[ds.kind.eq('generated')].delta_vina;dd=ds[ds.kind.eq('direct_baseline')].delta_vina
    lines.extend(['',f'生成组平均ΔVina为{gd.min():+.3f}至{gd.max():+.3f} kcal/mol，点估计显示相对种子的分数略有不利变化；直接检索基线为{dd.min():+.3f}至{dd.max():+.3f}。这与生成结果更接近种子结构的现象一致。四种方法没有呈现一个在两种条件下都占优的对接排序。'])
    lines.extend(['',
        'ΔVina=输出分数−同靶点种子分数，正值表示相对种子更不利，负值表示更有利。分数先在任务内平均，再在有有效分数的任务间等权平均；未交回或对接失败不填0。全部种子、150个确定性直接基线及其对接子集另有完整明细。这里呈现计算层面的亲和性取舍。','',
        '## 7. 数据和图怎样核对','',
        f'- 已从5000次成功回复重新抽取SMILES，并独立复算全部6000候选位置的有效性、属性、QED/SA、反馈和最终选择。2250条参考记录的属性与实际提示一致。',
        f'- 以原始结果JSON直接核对8组S/U和条件QED，均与汇总表一致。',
        f'- 在新的独立目录重算，{fresh["numeric_files_exact"]}个数值文件逐字节一致，包含绘图坐标。',
        f'- {sv["successful_jobs"]}个成功对接任务的实际SDF连接关系、输入中指定的立体信息、第一姿势Vina分数和能量矩阵均已核对。',
        '- 图的源表与文件指纹在figures/figure_index.json，实际坐标在figures/plotted_coordinates.csv。图01看S/U；图02看候选池与选择；图03看保存候选累计变化；图04看参考条件与结构分布；图05看结构门槛敏感性；图06看对接取舍。','',
        '人工检查时，先从properties/group_summary.csv选一组，再在properties/final_positions.csv按任务和重复定位；打开source_result所指JSON查看六个候选和最终分子。candidate_grid.csv的raw_request_path、raw_response_path直接定位完整上下文和原始回复。对接通过job_id打开docking/jobs/<job_id>/result.json、prepared_ligand.sdf和poses.pdbqt。','',
        '## 8. 费用、保存与后续','',
        f'生成5000次请求，无技术重试，按用量估算API费用{pv["estimated_usage_cost_cny"]:.2f}元，不含服务器租赁。生成后查询余额为{balance["balance_cny"]:.2f}元。本轮余额足够，没有触发余额不足暂停。',
        '余额不足保护已用无付费模拟验证：停止新请求、保存在途回复、恢复同一尝试、复用已成功轮次。真实运行中的环境兼容暂停也复用了已保存回复。完整生成进度另有ZIP归档和SQLite备份。','',
        '建议先与导师确认新增证据的表述边界，再整合论文：可以写新增LogP/TPSA条件下的可行性、结构保持和取舍；GIN保留为经过对照验证的可选后端。当前数据不支持把GIN写为优于Morgan或随机范例的性能创新，也不需要为追求显著性继续加跑。','',
        '远程完整实验目录：`/root/ProLiGen_single_property_1000_20261028/run_v2`。运行、恢复和各文件入口见根目录README_运行与数据指引.md。'])
    (ROOT/'analysis/结果与解读.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    sample=['logp_CHEMBL1741186_1_M','tpsa_CHEMBL1821_1_G',ff[ff.S.eq(0)].sort_values('slot_id').iloc[0].slot_id]
    dump(ROOT/'analysis/manual_sample_slots.json',dict(slots=sample,selection_rule='Two fixed pilot task/group examples plus lexically first failed slot; for source audit, not selected to illustrate method superiority.'))
    print('CHINESE RESULT REPORT WRITTEN',flush=True)


if __name__=='__main__':main()
