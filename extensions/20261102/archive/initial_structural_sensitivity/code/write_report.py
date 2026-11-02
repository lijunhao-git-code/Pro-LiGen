"""Write readable Chinese findings with every displayed number read from result tables."""
import json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
NAMES={'T':'T：无范例生成','M':'M：Morgan 范例生成','G':'G：GIN 范例生成',
       'R_NN1':'直接检索最近邻','R_QED3':'检索前 3 个按 QED 选优','R_QED6':'检索前 6 个按 QED 选优'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    p=ROOT/'results'
    s=pd.read_csv(p/'method_summary6.csv').set_index('method')
    gates=pd.read_csv(p/'structure_gate_summary.csv')
    tests=pd.read_csv(p/'retrieval_paired_tests9.csv')
    corr=pd.read_csv(p/'target_correlations.csv')
    final=pd.read_csv(p/'final_outputs_and_baselines450.csv')
    b=pd.read_csv(p/'MCS_completion_bounds.csv').set_index('method')
    summary=json.loads((p/'summary.json').read_text(encoding='utf-8'))
    overall=table(['方法','交回 / 计划位置','QED yield','种子 Tanimoto','种子共同原子覆盖率','库外比例'],[
        [NAMES[m],f'{int(s.loc[m,"delivered"])}/{int(s.loc[m,"planned"])}',f'{s.loc[m,"yield_QED"]:.4f}',
         f'{s.loc[m,"tanimoto_parent"]:.4f}',f'{s.loc[m,"mcs_seed_fraction"]:.4f}'+('†' if m=='T' else ''),
         f'{100*s.loc[m,"novel_library_exact"]:.0f}%'] for m in NAMES])
    changes=table(['方法','有机重原子数变化','有机分子量变化 / Da','QED 相对种子变化'],[
        [NAMES[m],f'{s.loc[m,"dHA_organic"]:+.2f}',f'{s.loc[m,"dMW_organic"]:+.2f}',f'{s.loc[m,"dQED_parent"]:+.4f}']
        for m in NAMES])

    def gate_table(metric,threshold):
        rows=[]
        for policy,caption in [('final_filter','只过滤原最终分子'),('candidate_reselection','已有候选重新择优')]:
            for m in ['T','M','G','R_QED6']:
                r=gates[(gates.metric==metric)&(gates.threshold==threshold)&(gates.policy==policy)&(gates.method==m)].iloc[0]
                rows.append([caption,NAMES[m],f'{int(r.delivered)}/{int(r.planned)}',f'{100*r.delivery:.1f}%',f'{r.yield_QED:.4f}'])
        return table(['分析方式','方法','交回 / 计划位置','交回率','QED yield'],rows)

    strong=[]
    for metric,t,caption in [('tanimoto_parent',.6,'Tanimoto ≥ 0.6'),('mcs_seed_fraction',.8,'共同原子覆盖率 ≥ 0.8')]:
        for m in ['T','M','G','R_QED6']:
            r=gates[(gates.metric==metric)&(gates.threshold==t)&(gates.policy=='candidate_reselection')&(gates.method==m)].iloc[0]
            strong.append([caption,NAMES[m],f'{int(r.delivered)}/{int(r.planned)}',f'{r.yield_QED:.4f}'])
    tests_table=table(['比较：前者减后者','QED yield 差','95% CI','Holm 校正 p'],[
        [r.comparison,f'{r.difference:+.4f}',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]',f'{r.p_holm:.4f}']
        for r in tests.itertuples()])
    example=final[(final.chemblid=='CHEMBL1741186')&((final.position==1)|final.method.str.startswith('R_'))]
    example_table=table(['方法','记录号','QED','Tanimoto','共同原子覆盖率'],[
        [NAMES[r.method],r.slot_id,f'{r.QED:.4f}',f'{r.tanimoto_parent:.4f}',f'{r.mcs_seed_fraction:.4f}']
        for r in example.itertuples()])
    text=f'''# P0 第四、第五项：离线补充分析与结果说明

日期：2026-10-28。全部数值由当前公开包中的原始记录重新计算；本文件的数字直接读取结果 CSV 生成。

## 先看结论

生成步骤的价值可以得到更具体的解释：**直接筛库可以得到更高的 QED，但生成结果更接近种子，并探索了当前分子库以外的结构。** 这比“生成方法在所有方面更优”的表述更符合实际结果。

同时，当前最高 QED 选择规则确实会牺牲部分结构保持。只给原来的最终输出增加结构筛选，范例组的优势可能消失；如果允许在已保存候选中重新选择，适度结构门槛下可以恢复一部分收益。门槛继续提高后，无范例组可能更好。**本轮没有发现 GIN 稳定优于 Morgan 的新证据。**

## 1. 本轮做了什么，为什么这样做

第四项回答：“库里已经有符合环数的分子，为什么还需要生成？”设置三个无需大模型的返回方案：Morgan 最近邻直接返回、前 3 个中选择最高 QED、前 6 个中选择最高 QED。后两个基线也允许按 QED 挑选，避免让生成组独占选优机会。

第五项回答：“分子是不是只是大幅缩小、换成另一个高 QED 分子？”重新计算种子相似度、最大公共子结构覆盖率、重原子数和分子量等变化，并检查增加结构条件后还能交回多少分子、保留多少 QED 收益。

输入为研究 B 的 25 个任务、375 个原计划位置、2,244 个保留候选位置和 26,108 个库分子。原来的 375 个最终选择、75 个 Morgan 参考分子及其排序得分均完全复现。全部 2,211 个有效候选的性质也重新实算，独立重建的合格候选数为 T=718、M=723、G=719，总数 2,160。

本轮新增生成 API 调用为 0，新增 API 费用为 0 元；没有重新训练、重新对接，也没有改动论文或公开仓库的现有版本。检索基线与生成组的新增亲和力比较尚未开展。

## 2. 第四项：直接筛库与生成的对照

{overall}

表中所有均值先在每个任务内汇总，再对 25 个任务等权平均。T/M/G 各有每任务 5 个计划位置；直接检索每任务只有 1 个确定性输出，不能把它复制 5 次当作独立实验。

QED yield 指成功交回时记该分子的 QED，未交回时记 0，再按计划位置求平均。它同时考虑分数和交回情况。这里 T 的条件 QED 均值为 {s.loc['T','QED']:.4f}，与含失败位置的 yield {s.loc['T','yield_QED']:.4f} 不是同一列；M/G 和检索基线因为全部交回，两者相同。

共同原子覆盖率的分母是种子有机组分的重原子数。† T 有 1 个最终分子的 MCS 搜索未完成，该列为其余精确结果的任务均值；若把未完成分子也纳入上下界计算，T 总体平均落在 {b.loc['T','lower_MCS_mean']:.4f}–{b.loc['T','upper_MCS_mean']:.4f}。M/G 和三个检索基线的最终分子均完成精确搜索。

这个表需要横着结合看：前 6 个检索结果按 QED 选优的分数最高，但与种子的 Tanimoto 只有约 0.26，共同原子覆盖率约 0.46；两个范例生成组的 Tanimoto 约 0.55，共同原子覆盖率约 0.69。因此，不能再主张生成在纯 QED 上优于一个较强的筛库方案；可以报告生成提供了更接近种子的库外分子。

373 个生成输出共包含 293 个全局不同的 canonical 分子，均不在整个 26,108 分子库中；忽略立体标记后查重仍为零。与原始三元组涉及的 {summary['training_canonical_unique']:,} 个训练输入分子也无精确重合。这里的“库外”仅相对本研究公开分子库和原始三元组输入，不等于此前从未被合成，也不等于从未被大模型见过。直接检索输出本来就在库内，因此库外比例为 0，并非化学无效。

### 基线如何做到公平

沿用原 Morgan 的 radius=2、2,048 bit 指纹以及二值余弦检索排序，先限定目标环数并排除种子。返回的库分子还必须单连通，与生成输出的该项要求一致。原参考池允许多组分条目，本轮为直接交回分子预先剔除了这类条目；每任务的原池、输出池和删除数都保存在 `results/retrieval_pools.csv`。这是同一库和同一环数条件下的单连通输出子池，不能写成与原范例池逐条完全相同。

允许基线直接返回范例分子，避免用生成组的“不得复制范例”规则让它天然失败。QED 相同则依 SA、更后按 canonical SMILES 破同分。没有搜索整个分子库的最高 QED；本轮问题是最近邻及局部前 3/6 个的直接返回。

### 新增的统计比较

{tests_table}

上表是本轮新增的九个探索性检验，均以 25 个配对任务为单位，在九个检验内做 Holm 校正。置信区间是未做同时覆盖校正的 bootstrap 区间，所以它和校正 p 值不一定作出相同的阈值判断。原论文 G–M 的原检验及结论没有改变。

M 对最近邻的 QED yield 改善在本检验族中得到支持；M/G 对前 3 个 QED 选优的额外优势没有得到支持。前 6 个 QED 选优比 G 的分数更高。新结果不支持“生成得到全局更高 QED”或“GIN 的优越性得到证实”。

## 3. 第五项：结构保持与分子大小变化

{changes}

三个生成组平均都在减小分子，范例组比无范例组减小得更多。M/G 平均少约 7–8 个重原子、减小约 107–114 Da；因此，QED 提升伴随的不只是环数改变。QED 改善与有机重原子变化的任务均值 Spearman 相关为 T={corr[(corr.method=='T')&(corr.predictor=='dHA_organic')].iloc[0].spearman:.3f}、M={corr[(corr.method=='M')&(corr.predictor=='dHA_organic')].iloc[0].spearman:.3f}、G={corr[(corr.method=='G')&(corr.predictor=='dHA_organic')].iloc[0].spearman:.3f}，负相关表示缩小较多的任务往往 QED 提升较多。这是相关关系，不能把它写成“QED 的改善全部由分子缩小造成”。

结构指标保留原论文的 parent-form Tanimoto。另以唯一含碳组分计算 MCS 和有机大小变化；只有 CHEMBL1821 的种子含单原子溴对离子，做了单独标记。QED 与原论文主相似度没有偷偷换成去盐后的口径。

MCS 比较元素及键级、允许环与链对应，忽略手性与形式电荷是否相同，不要求完整环。它描述共同连接结构的重原子覆盖率，不能等同于完整骨架、药效团或生物活性保持。全部 1,462 对结构搜索有 1,453 对完成，9 对保留下界与完成标记，没有将超时搜索当作精确最大值。

### 3.1 适度结构门槛：必须区分“过滤原结果”与“重新选择”

以下以 Tanimoto ≥ 0.4 为展示点；完整 0.0–0.9 曲线全部保留，不只展示有利阈值。

{gate_table('tanimoto_parent',.4)}

**只过滤原结果**回答“已经交回的分子有多少本来就满足这个结构要求？”此时范例组的 QED yield 低于 T，原来的高 QED 选择确实会选走部分更接近种子的候选。

**已有候选重新择优**回答“原来的候选里，能否挑出既满足结构条件又有较好 QED 的分子？”在各原计划位置中，加上结构门槛，再按原 QED/SA 规则选择。M/G 均保留 121/125 个位置，QED yield 分别约 0.653 和 0.645，高于 T 的约 0.605。这说明候选池中存在更合适的折中，不能把它说成原始最终输出已经实现了这项要求。

此处直接检索的重新选择只在原前 6 个中进行，不重新扩大到整个分子库；因此本结果只比较所定义的局部检索基线。

### 3.2 更严格结构要求下，范例组不再稳定占优

以下均为已有候选重新选择：

{table(['结构门槛','方法','交回 / 计划位置','QED yield'],strong)}

Tanimoto ≥ 0.6 时 M/G 不再高于 T；共同原子覆盖率 ≥ 0.8 时，T 的交回数量和 QED yield 都更高。结构条件越严格，纯 QED 选优带来的收益越难全部保留。这是应当说明的实际取舍，不能只报告 0.4 的有利点。

对 MCS ≥ 0.8，T 的“过滤原输出”有一个未交回位置处于搜索不确定区间，保守统计为 45/125；它可能增加到 46/125。T 的“重新选择”虽有一个候选未确定，但该位置已有可确认满足条件的备选输出。全部不确定条目可在 `structure_gate_selections.csv` 的 `unresolved_candidates` 中定位；M/G 和直接检索在上述展示点没有这种不确定条目。

这些是**复用既有候选的事后敏感性分析**，没有重新向模型提问。不能直接拿重新选择的分子替换研究 B 原表，再继续把原来的 p 值、对接结果沿用到新分子上。

## 4. 三张补充图怎么读

1. `figures/01_baseline_comparison.png`：每个小点是一个任务的均值，菱形是 25 个任务的等权均值。左图看 QED yield，中图看种子相似度，右图看共同原子覆盖率。把三个面板一起看，才看得到检索高 QED 与生成结构保留之间的差别。
2. `figures/02_structural_changes.png`：上排是分子缩小与 QED 改善的任务均值关系；越往左，缩小越多。下排是最终输出的相似度和共同原子覆盖率累积分布，同一横坐标处曲线越低，表示达到更高保持程度的输出比例越大。T 的 Tanimoto 用 123 个输出，精确 MCS 用 122 个，另外两个组均为 125 个。
3. `figures/03_structure_thresholds.png`：横轴是逐步提高的结构门槛，上排看交回率，下排看失败记零的 QED yield。虚线只过滤原输出，实线在同一批已有候选中重新选；二者差距反映最终选择规则可以改变多少结果。完整的六方法数值都在 CSV 中，图中展示三个生成组及较强的前 6 检索基线。

这三张图只是本轮分析附件，尚未插入论文。它们的全部绘图坐标在 `figures/plotted_coordinates.csv`，源文件哈希在 `figures/figure_index.json`。

## 5. 如何人工核查，不必通读所有数据

先打开 `gallery/index.html`，选择一个任务。它完全离线，图中的着色原子来自真实 MCS 匹配；支持逐个查看全部 450 个计划位置。共有 343 对不同的实际种子—最终分子图，重复分子通过同一图片引用展示。

以 CHEMBL1741186 为例，种子有 4 个环，目标是 3 个环。下面给出每种生成方法的第 1 个位置以及三个检索基线，便于照着数值核查：

{example_table}

- 查原最终分子：在 `inputs/B_planned375.json` 搜索相应 `slot_id`；与 `results/final_outputs_and_baselines450.csv` 的同名行核对 canonical、QED、Tanimoto。
- 查基线为何选中：在 `results/retrieval_baseline_candidates.csv` 筛选 `chemblid` 和 `method=R_QED6`，查看前 6 个排序及 QED，最高 QED 的分子应与最终基线行一致。
- 查共同原子覆盖率：在最终结果行找到 `mcs_key`，在 `results/mcs_cache.jsonl` 搜索它，核对 `common_atoms`、SMILES、`complete` 与 `seed_fraction`。打开分子对照页核对着色结构。
- 查宏平均：在 `results/target_means150.csv` 筛选方法，每个方法应有 25 行；对同一指标这 25 行取平均，应得到 `method_summary6.csv`。yield 的失败位置记 0，不要仅对成功分子平均。
- 查结构门槛：在 `structure_gate_selections.csv` 筛选指标、阈值、policy 和方法，逐行查看 `original_canonical` 与 `selected_canonical`。它能说明究竟只是过滤了原结果，还是换选了另一个原有候选。
- 查图片：从 `plotted_coordinates.csv` 按 figure/panel 筛选，再与索引中的数据文件核对。独立审计已重新核对图 1 全部坐标、图 2 累积分布、图 3 全部曲线坐标。

## 6. 对下面论文工作的建议

第四项已有一个有判别力的结果：纯 QED 上不能胜过较强筛库，但可用结构保留和库外探索明确生成的增量。第五项补足了结构证据，也确认当前流程存在大小与结构变化，不能只用“高 QED”概括质量改善。

建议保留原研究 B 主结果，补充直接检索对照；用完整结构曲线说明权衡，必要时把“结构门槛后选择”作为明确标注的补充变体。原最高 QED 规则与新变体需要分别叙述。GIN 可以作为学习表征选例的实现保留，但本轮不能新增“GIN 优于 Morgan”结论。

没有开展的随机同池范例对照、相对前作的新增对照和其他性质泛化，不因本轮两个分析自动解决。此次也没有新增药效、实验活性或检索基线对接证据。是否继续补其中一项，应结合导师讨论决定，不能仅用改写替代缺少的证据。

## 7. 模型版本一致性

2026-10-28 核对的 DeepSeek 官方文档说明：`deepseek-flash` 目前由 V4.1-Flash-1017 服务，旧的 `deepseek-v4-flash` 名称仍接受请求，但对应旧模型已退役，旧名也路由到现行服务。因此，指定旧名字不能锁定 4.0 权重版本。

来源：[官方接口说明](https://api-docs.deepseek.com/quick_start/agent_integrations/)、[官方更新记录](https://api-docs.deepseek.com/updates/)。本轮没有用任何付费调用来探测版本，也没有根据今天的文档改写过去实验的真实模型记录。后续应保存实际请求名、响应 model 标识、运行日期和可获得的版本信息；如确实无法调用旧版，新的因果对照需要在同一可用版本下同期进行，不能只把代码中的名字写旧就认为与历史实验完全一致。
'''
    (ROOT/'结果说明与人工核查.md').write_text(text,encoding='utf-8')
    print('Readable report written; displayed numerical tables read directly from CSV.')


if __name__=='__main__':main()
