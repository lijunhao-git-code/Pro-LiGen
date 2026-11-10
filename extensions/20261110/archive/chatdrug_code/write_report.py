"""A concise report with numeric values read only from this run's saved analysis."""
import json
from common import ROOT,read,now
import pandas as pd

def n(x,d=4):return 'NA' if pd.isna(x) else f'{x:.{d}f}'
def main():
 summary=pd.read_csv(ROOT/'analysis/native_summary6.csv');tests=pd.read_csv(ROOT/'analysis/paired_tests6.csv')
 replay=pd.read_csv(ROOT/'analysis/candidate_replay_summary.csv');resources=read(ROOT/'analysis/API_resource_summary.json')
 names={'ring':'恰好减一环','logp':'LogP下降1.0±0.2','tpsa':'TPSA下降20±5 Å²'}
 lines=['# ChatDrug与Pro-LiGen三属性对照结果','本轮750个正式位置，另有12个独立技术试跑位置。原论文未修改，未进行训练或对接。',
  '## 实验比较什么',
  '沿用25组靶点与种子，分别完成三种单属性编辑；每种方法、每个任务与条件运行5次。Pro-LiGen G与ChatDrug适配版各375个正式位置。',
  '双方使用同一DeepSeek官方服务、相同可用属性合格池和共同的冻结背景。背景取自归档的前三轮靶点／文献／种子分析，按任务与条件固定；本轮所有提案与修订回复均为新生成，历史候选没有混入结果。',
  'G保留两轮各3候选、实算反馈、跨轮保留和最高QED选择。ChatDrug保留初始5个提案只解析首个用于原生判断、未达标时检索相似合格范例并修订、首次达标即返回；最多初始加2次修订。它不是把DeepSeek直接换名，任务条件、共同输入和外部合格检查属于已记录的适配。',
  'G最多明确请求6个候选；ChatDrug最多请求5+1+1个，但其原生流程每轮只判断首个。候选数量与最终选择规则的差异属于流程比较的一部分，下面的共同选择重放用于进一步解释。',
  '## 两个流程实际交回了什么',
  '| 条件 | 方法 | 达标／125 | 达标率 | QED产出U | 达标输出QED | SA原始分数 | 种子Tanimoto |',
  '| --- | --- | --- | --- | --- | --- | --- | --- |']
 for _,r in summary.iterrows():lines.append(f'| {names[r.property]} | {"Pro-LiGen G" if r.group=="G" else "ChatDrug适配版"} | {int(r.delivered)}/125 | {r.S*100:.1f}% | {n(r.U)} | {n(r.conditional_QED)} | {n(r.conditional_SA_raw)} | {n(r.conditional_tanimoto)} |')
 lines+=['达标率看能否交回合格分子；U将每个未交付位置记0，综合反映交付与QED。条件QED、SA和Tanimoto只描述已交付分子，按任务汇总后平均；SA越低越好，Tanimoto越高表示更接近种子。',
  '## 按25个配对任务比较',
  '每个任务先汇总5次结果，再比较两种方法。下表为G减ChatDrug，正值表示G更高；95%区间来自任务bootstrap，p值对三条件×S/U共6项主要比较作Holm校正。',
  '| 条件 | 指标 | G−ChatDrug | 95%区间 | Holm p |',
  '| --- | --- | --- | --- | --- |']
 for _,r in tests.iterrows():
  if r.metric=='S':diff=f'{r.difference*100:+.2f}个百分点';ci=f'[{r.ci_low*100:+.2f}, {r.ci_high*100:+.2f}]个百分点'
  else:diff=f'{r.difference:+.4f}';ci=f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]'
  lines.append(f'| {names[r.property]} | {r.metric} | {diff} | {ci} | {r.p_Holm:.5g} |')
 lines+=['## 共同选择规则说明了什么',
  '将本轮保存的候选按相同规则离线重放，不额外生成分子。它回答“如果换一种最终选择方式，已提出的分子能交回什么”，不能替代真实对话策略的重新运行。',
  '| 条件 | 方法 | 原生S | 原生U | 共同最高QED选择S | 共同最高QED选择U |',
  '| --- | --- | --- | --- | --- | --- |']
 for prop in ['ring','logp','tpsa']:
  for g in ['G','CD']:
   sub=replay[(replay.property==prop)&(replay.group==g)].set_index('policy');native=sub.loc['native'];best=sub.loc['best_QED']
   lines.append(f'| {names[prop]} | {"Pro-LiGen G" if g=="G" else "ChatDrug适配版"} | {native.S:.3f} | {native.U:.4f} | {best.S:.3f} | {best.U:.4f} |')
 lines+=['若ChatDrug在共同选择下明显改善，说明其原生选择／停止规则是结果差异的重要来源；不能将全部差异归给模型提出分子的能力。`candidate_frontier_summary.csv`进一步将双方限定为前k个已记录候选，展示共同选择下的交付与QED产出。',
  '本轮比较的是保留GIN的完整框架与另一种流程，结果不能单独证明GIN优于Morgan；原T/M/G/R关于选例后端的结论仍需依据原消融结果。',
  '## 实际调用与资源记录']
 for r in resources:lines.append(f'- {"正式实验" if r["cohort"]=="formal" else "独立试跑"}：{r["submissions"]}次实际提交，{r["successful_calls"]}次成功调用；按归档费率和usage估计{r["archived_ratecard_estimate_cny"]:.2f}元，全部按归档高峰费率估计{r["archived_peak_rate_estimate_cny"]:.2f}元；单次调用耗时中位数{r["median_call_seconds"]:.1f}秒，首末新调用跨度{r["wall_clock_seconds"]/60:.1f}分钟。')
 lines+=['费用为本轮已保存usage的估计，不是充值金额或账户其他调用的花费；冻结背景的历史获取费用未计入本轮。模型返回标识与服务指纹见`API_resource_summary.json`，不据此宣称独立核验了供应商底层权重。',
  '## 如何核对数据与图片',
  '1. 看`analysis/native_summary6.csv`的六行汇总；每行固定125个正式位置。',
  '2. 到`analysis/final_outputs750.csv`按属性、方法筛选，查看每次是否交回分子、性质以及`response_source`对应结果文件。',
  '3. 打开对应`results/位置.json`，核对`selected_candidate`、`candidates`、`trace`；再到`responses/位置_a轮次_t对话轮_s提交次数.json`查看原始模型回复，`requests/`保存对应提示。',
  '4. `analysis/task_means150.csv`是25任务×3条件×2方法的统计单位，`paired_tests6.csv`给出六项主要检验。',
  '5. `figures/native_comparison.png`使用`native_summary6.csv`和`paired_tests6.csv`；`common_selector_frontier.png`使用`candidate_frontier_summary.csv`。每个图的数据来源和文件校验值另见`figure_data_sources.json`。',
  '6. `analysis/full_audit.json`记录从原始回复重新解析、独立RDKit性质复算、检索与选择核查；`protocol.json`和`inputs/generation_locked.json`记录事先固定的设置与输入。',
  '需要修改论文时，先讨论原生结果与共同选择重放是否支持拟写结论；本轮没有自动写入论文或上传公开仓库。']
 audit=read(ROOT/'analysis/full_audit.json')
 lines.insert(lines.index('## 实际调用与资源记录')+1,f'有{audit["retried_positions"]}个位置因回复截断等技术原因按预定上限重试；截断回复未用于最终输出，全部调用与费用仍计入资源记录。未对性质不达标进行质量重试。')
 (ROOT/'实验结果与数据核对.md').write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
 print(json.dumps(dict(status='report_written',lines=len(lines))))
if __name__=='__main__':main()
