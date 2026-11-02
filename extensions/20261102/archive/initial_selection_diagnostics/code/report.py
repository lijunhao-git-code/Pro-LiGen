"""Readable Chinese findings; every numerical table comes from computed CSV records."""
import json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NAMES={'T':'无范例T','M':'Morgan范例M','G':'GIN范例G'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    r=ROOT/'results';s=pd.read_csv(r/'policy_summary.csv');q=pd.read_csv(r/'candidate_quality_summary3.csv').set_index('group')
    tests=pd.read_csv(r/'group_comparisons6.csv');effects=pd.read_csv(r/'selection_effects6.csv')
    full=s[(s.k==6)&s.policy.isin(['first_eligible','random_unique_expected','best_QED'])].set_index(['group','policy'])
    policies=['first_eligible','random_unique_expected','best_QED']
    main_table=table(['方法','首个合格：QED yield','随机合格：期望QED yield','最高QED：QED yield'],[
        [NAMES[g]]+[f'{full.loc[(g,p),"yield_QED"]:.4f}' for p in policies] for g in NAMES])
    quality_table=table(['方法','有效候选 / 750计划位置','全部有效候选的平均QED','全部合格候选的平均QED','有效候选QED贡献 / 计划位置'],[
        [NAMES[g],f'{int(q.loc[g,"valid_total"])}/750',f'{q.loc[g,"valid_QED"]:.4f}',
         f'{q.loc[g,"eligible_QED"]:.4f}',f'{q.loc[g,"valid_proposal_QED_yield"]:.4f}'] for g in NAMES])
    inference=table(['比较','随机选择期望yield差','未校正95% CI','六检验内Holm p'],[
        [row.comparison,f'{row.difference:+.4f}',f'[{row.ci_low:+.4f}, {row.ci_high:+.4f}]',f'{row.p_holm:.5f}']
        for row in tests[tests.outcome=='random_unique_expected_yield'].itertuples()])
    structural=table(['方法','首个合格：Tanimoto','随机合格期望：Tanimoto','最高QED：Tanimoto'],[
        [NAMES[g]]+[f'{full.loc[(g,p),"tanimoto"]:.4f}' for p in policies] for g in NAMES])
    gains=table(['方法','最高QED相对随机选择的yield增加','未校正95% CI'],[
        [NAMES[row.group],f'{row.difference:+.4f}',f'[{row.ci_low:.4f}, {row.ci_high:.4f}]']
        for row in effects[effects.comparison=='best_QED-random_unique_expected'].itertuples()])
    example=next(json.loads(line) for line in (r/'random_unique_support2250.jsonl').read_text(encoding='utf-8').splitlines()
                 if json.loads(line)['slot_id']=='CHEMBL1741186_1_M' and json.loads(line)['k']==6)
    weights=', '.join(f'{x:.4f}' for x in example['QED'])
    text=f'''# QED选择与评价：已有候选的离线审计

2026-10-28。结果来自原研究B的25个任务、375个计划运行位置和2,244个已保存候选位置。没有新增生成、训练或对接，新增API费用为0元。

## 这次检查得到什么

**原来的QED评价可以保留。最高QED选择确实提高了最终分数，但范例组的QED收益并不完全依赖最后的最高分选择。** 随机从合格候选中选一个时，Morgan和GIN组相对无范例组仍有更高的QED yield；全部有效候选的平均QED也更高。GIN相对Morgan的额外收益仍未得到确认。

这为审稿意见提供了直接回应：原主表评价完整流程最终交回的分子，本轮新增候选层面和替代选择规则结果，说明最终高分是如何形成的。

## 1. 选择规则重放

每个原计划位置最多有两轮、每轮三个候选。沿用原来的有效性、单连通、恰好减一环以及不得精确复制种子/所给范例的检查。对同一批已保存候选重放：

- 首个合格：按原来的轮次、位置顺序，选择最早的合格候选。
- 随机合格：对该位置不同的合格canonical分子均匀抽取。主要报告其精确期望值，不依赖某次随机抽取运气。
- 最高QED：按原部署规则先保留每个canonical最后一条合格记录，按QED降序、SA升序、canonical升序选择。

{main_table}

QED yield在成功交回时记该分子的QED，没有可交回分子时记0。先在每个任务的五个计划位置内平均，再对25个任务等权平均。三种选择规则可交回的位置数完全相同：T为123/125，M/G为125/125；表中区别来自选中了谁，而不是删掉了失败位置。

“随机合格”不是此前计划中的随机范例R组：这里随机选择最终输出；R组随机选择提供给模型的输入范例。本轮没有运行新的R组。

### 随机选择怎样得到一个精确数值

例如`CHEMBL1741186_1_M`，六个已保存位置内有{len(example['canonical'])}个不同的合格结构，其QED依次为：{weights}。均匀随机选一个时，每个结构概率都是1/{len(example['canonical'])}，期望QED为这些QED的算术平均，即{sum(example['QED'])/len(example['QED']):.4f}。

这个期望数值不是构造了一个“平均分子”，而是该随机选择规则的平均表现。`random_unique_support2250.jsonl`记录每个位置、每个候选数量下的实际SMILES、QED和抽样概率。此外还保存了一套以42为基础种子的具体随机选择，以及“按候选出现次数随机抽取”的敏感性结果。

### 不挑最高分，范例组是否仍有收益

{inference}

M−T和G−T在这组新增探索性检验中得到支持；G−M没有得到支持。这说明当前保留候选中，范例组的合格候选平均QED收益仍然存在，并非只在挑最高分后才出现。这里的差异只对应QED相关结果，不等于全部药物性质或全部分子都更好。

检验以25个配对任务为单位，使用10,000次任务bootstrap和100,000次配对符号翻转。新增检验共六个：上表三项，以及全部有效候选QED贡献的三项组间比较，统一在六项内做Holm校正。CI是未做同时覆盖校正的95%区间。它们是本轮事后探索性分析，未改动原论文G−M等预设检验的身份或结果。

## 2. 最后选最高QED，额外增加多少分数

{gains}

在固定候选池上，最高QED相对随机选择额外提高约0.07–0.09的yield，程序选优的作用确实存在。这两个事实可以同时成立：**范例组提供了平均QED较高的候选，最高QED选择进一步提高了最终交回分数。**

最佳值不会低于均值是选择规则的机械性质，因此此处报告差值及描述性bootstrap区间，不用显著性检验证明这一机械关系。也不能据此把原方法的收益按比例精确分成“多少属于生成、多少属于筛选”：本轮固定了原来已发生的反馈和生成轨迹，只更换最终选择规则。

## 3. 候选整体的QED怎样

{quality_table}

“有效”指RDKit能够解析；“合格”还要求满足环数、单连通与反照抄条件。这两列QED都没有经过最终最高QED选择，均先在每个任务的候选位置中平均，再对任务等权平均。保留重复出现的提案，因为这里评价每个提案位置。

三组有效率接近，M/G的全部有效候选平均QED仍高于T，因此差别不只是删掉几个无效结果造成的。将无效和缺失位置记0、按每任务30个计划候选位置计算QED贡献后，M−T为{tests[(tests.outcome=='valid_proposal_QED_yield')&(tests.comparison=='M-T')].iloc[0].difference:+.4f}，Holm p={tests[(tests.outcome=='valid_proposal_QED_yield')&(tests.comparison=='M-T')].iloc[0].p_holm:.5f}；G−T为{tests[(tests.outcome=='valid_proposal_QED_yield')&(tests.comparison=='G-T')].iloc[0].difference:+.4f}，Holm p={tests[(tests.outcome=='valid_proposal_QED_yield')&(tests.comparison=='G-T')].iloc[0].p_holm:.5f}。

这支持候选层面的平均QED有所改善，不要求每个任务都获益，也不意味着两组QED分布在所有阈值上都严格占优。完整分布和任务点保留在图2。

### 提高QED也改变了结构相似度

{structural}

三组从首个合格候选改成最高QED候选后，平均种子相似度均下降。改成随机选择时，相似度也高于最高QED选择。这与前一轮结构保持审计一致：当前选择规则倾向于交回QED更高、同时结构改变较大的候选。因此，应把QED与结构保持结合讨论，不能把一个高QED分子直接概括为所有方面质量更好。

## 4. 候选数量曲线能说明什么

本轮把每条原轨迹按顺序取前1至6个候选位置，分别执行首个、随机、最高QED选择。无效和缺失候选占用其原位置，没有事后只挑有效候选组成“前k个”。

最高QED曲线随着可检查候选增加而提高；随机均值不保证单调，因为新增候选也可能低于已有均值。第4个位置开始来自第二轮，此轮在原实验中已经看过完整的第一轮程序反馈，所以曲线的阶段变化同时包含了候选来源的变化。

**这张图是“现有候选数量的敏感性”，不是重新按不同API费用、不同候选请求数运行的实验。** 每轮三个候选来自同一次生成请求，不能把只检查一个位置等同于只花了原来三分之一的钱。也不能只靠第二轮曲线上升就断言反馈本身有因果收益；原研究C负责检验额外数值反馈。

## 5. 这条审稿问题怎样回应

可以保留QED与QED yield作为完整流程的输出指标，同时补充以下证据：固定候选下首个/随机/最高QED重放、全部候选的质量分布，以及候选数量敏感性。现有结果支持“范例组的候选平均QED提高，最终选优进一步提高输出QED”。原来的最终输出结果不是算错或失效。

适合后续写入论文的英文表述为：

> On the retained Study-B candidate pools, both example-guided configurations retained a higher expected QED yield than the no-example configuration under uniform selection among eligible unique structures. Candidate-level analyses also showed higher mean QED before final selection. QED-based selection further increased the delivered yield, accompanied by lower similarity to the seed. An additional benefit of GIN over Morgan was not established.

此段对应本轮事后重分析，不能改写成新一轮独立生成的结果。随机输入范例是否同样有用、GIN是否优于Morgan、向其他属性的迁移仍分别依赖对应实验。本轮只完成当前选择与评价问题，不启动新属性生成。

## 6. 图片和人工核查入口

- `figures/01_selection_policies.png`：左边读三种选择规则的QED yield，右边看结构相似度的变化。误差线为各方法均值的bootstrap区间；配对组间差异应看`group_comparisons6.csv`，不能仅凭两条均值误差线是否重叠判断显著性。
- `figures/02_candidate_quality.png`：上排每个点代表一个任务未经过最终选优的候选平均QED；菱形为25任务均值。下排是提案位置的描述性累积分布，不能把数百个相关候选当作数百个独立统计样本。
- `figures/03_candidate_count.png`：从前1到前6个已保存位置重放。上排交回率在三种选择规则间相同，下排显示最终选择如何影响yield；竖线右侧开始纳入反馈后的第二轮位置。

人工核查可从`CHEMBL1741186_1_M`开始：

1. 在`candidate_grid2250.csv`筛选slot_id，核对六条原始SMILES、valid、eligible、QED、轮次和位置。
2. 在`random_unique_support2250.jsonl`搜索同一slot_id并找到k=6，去重后的每个结构概率应相等，总概率为1；平均QED应与`selection_by_slot12000.csv`的random_unique_expected行一致。
3. 在`selection_by_slot12000.csv`筛选同一位置和k=6，最高QED行应选中合格结构中QED最高者；first行应选中时间顺序最早者；固定种子random_unique_draw42行应属于该合格集合。
4. 每组每规则有125条计划运行位置。先对五条同任务记录平均，再对25个任务平均，应得到`policy_summary.csv`。
5. `figures/plotted_coordinates.csv`给出全部4611条绘图数据记录；`figure_index.json`给出源CSV路径和SHA-256。独立审计逐一核对了均值与累积分布坐标。

核心复核结果：375个原最终分子、1500条原重放记录完全复现；全部有效候选的性质和合格判定重新实算。另以独立算式核对9000条规则期望值、2250个随机分布及计划分母。原论文和原公开包的完整文件指纹另行核验。
'''
    (ROOT/'结果说明与人工核查.md').write_text(text,encoding='utf-8')
    print('Chinese findings and manual verification guide written from result CSVs.')


if __name__=='__main__':main()
