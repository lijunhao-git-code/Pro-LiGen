"""Original Chinese request builders extracted from the frozen experiment.
Call design_prompt(entry) and optimize_prompt(entry), then append feedback(rows,entry).
The entry schema is documented in README.md. This module makes no API requests.
"""
import json
PROPERTIES={'logp':dict(label='Crippen LogP',delta=-1.,tolerance=.2),'tpsa':dict(label='TPSA',delta=-20.,tolerance=5.,units='Angstrom squared')}
def properties(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid SMILES')
    return dict(logp=float(Descriptors.MolLogP(mol)),tpsa=float(Descriptors.TPSA(mol)))
def design_prompt(entry):
    prop=entry['property'];info=PROPERTIES[prop]
    refs=''
    if entry['exemplars']:
        refs='- 满足相同属性目标区间的参考分子（共3个）：\n'+'\n'.join(
            f'  - Exemplar {i}（{info["label"]}={properties(s)[prop]:.8f}）：{s}' for i,s in enumerate(entry['exemplars'],1))+'\n'
        approach='分析参考分子相对于种子的结构改造方式，结合种子及靶点信息进行设计。'
    else:approach='分析种子的结构改造方式，结合种子及靶点信息进行设计。'
    return f'''你是一位经验丰富的药物化学专家，擅长启发式分子设计。
【任务】
1. {approach}
2. 生成3个全新的、化学上有效的SMILES，按当前单属性条件进行温和改造，并尽量保留种子的关键结构。
3. 只设计单个连通且含碳的分子，不要直接输出种子或输入中提供的参考分子。
【输入信息】
- 靶点UniProt ID：{entry['uniprotid']}
- 靶点ChEMBL ID：{entry['chemblid']}
- 种子SMILES：{entry['seed_smiles']}
- 种子{info['label']}：{entry['seed_property']:.8f}
{refs}【目标属性】
- 当前唯一性质硬条件：{info['label']}相对种子降低{abs(info['delta']):g}，允许误差±{info['tolerance']:g}。
- 因此每个输出的RDKit实算{info['label']}应位于[{entry['target_lower']:.8f}, {entry['target_upper']:.8f}]（含端点）。
- 不额外要求环数、分子量或另一属性达到某个数值。
【输出格式】
## Structural Analysis
用1至2句话说明改造策略。
## Generated Molecules
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>
仅输出3个带标签候选。不要编造QED、SA、Vina或实验活性，也不要声称已实际对接。'''

def optimize_prompt(entry):
    info=PROPERTIES[entry['property']]
    return f'''依据前文靶点、文献、种子以及程序实算结果，进行一次分子优化。
唯一性质硬条件：{info['label']}应位于[{entry['target_lower']:.8f}, {entry['target_upper']:.8f}]。
生成单连通含碳分子，不复制种子或输入范例；在达标前提下提高类药性并注意合成可行性和结构保留。
程序会合并两轮各3个候选：按实际QED从高到低，完全同分时按SA原始分数从低到高，再按canonical SMILES排序。
只提出3个候选，不声称自行执行RDKit、QSAR或对接，不编造性质和结合结果。
简短解释后只输出：
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>'''

def feedback(rows,entry):
    fields=['candidate_position','canonical','valid','property_value','delta_property','target_error','property_hit','eligible','reason','QED','SA_raw']
    actual=[{k:r.get(k) for k in fields} for r in rows]
    return ('\n【程序实际计算的上一轮结果】\n'+json.dumps(actual,ensure_ascii=False)+
        f'\n当前目标仅为{PROPERTIES[entry["property"]]["label"]}位于[{entry["target_lower"]:.8f}, {entry["target_upper"]:.8f}]。'
        '\n以上由RDKit与标准SA scorer计算。上一轮合格候选会保留；请再提出3个候选，不复制种子或所给范例。未执行对接，不要编造Vina或实验结果。')
