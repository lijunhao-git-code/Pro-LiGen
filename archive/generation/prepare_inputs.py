"""Freeze T/M/G inputs before paid generation. No training, no LLM generation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import importlib.metadata
import json
import random
import re
import shutil
import inspect
import sys
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch_geometric.data import Batch
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from common import PROJECT, EXP, OUT, sha, dump, now, read_jsonl, canon, properties, scorer

sys.path.insert(0, str(PROJECT))
from run import GINEncoder, mol_to_graph
FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def embed(model, graphs, device):
    result = []
    with torch.inference_mode():
        for offset in range(0, len(graphs), 128):
            data = Batch.from_data_list(graphs[offset:offset + 128]).to(device)
            result.append(F.normalize(model(data), dim=1).cpu().numpy())
    return np.concatenate(result)

def topk(scores, eligible, keys):
    idx = np.flatnonzero(eligible & np.isfinite(scores))
    if len(idx) < 3:
        raise RuntimeError('Fewer than 3 exact-condition unique candidates')
    cutoff = np.partition(scores[idx], -3)[-3]
    idx = idx[scores[idx] >= cutoff]
    return idx[np.lexsort((keys[idx], -scores[idx]))[:3]]

def property_turn(info, exemplars):
    reference = ''
    if exemplars is not None:
        reference = '- 满足相同环数条件的参考分子（共3个）:\n' + '\n'.join(
            f'  - Exemplar {i}（环数={info["target_rc"]}）: {smi}' for i, smi in enumerate(exemplars, 1)) + '\n'
        approach = '分析参考分子相对于种子的结构改造方式，结合种子及靶点信息进行设计。'
    else:
        approach = '分析种子的结构改造方式，结合种子及靶点信息进行设计。'
    return f'''你是一位经验丰富的药物化学专家，擅长启发式分子设计。

【任务】
给定目标靶点及其种子配体，请你：
1. {approach}
2. 生成3个全新的、化学上有效的 SMILES 字符串；温和简化一个环系，同时尽量保留关键骨架与口袋接触。
3. 不要直接输出种子配体或输入中提供的参考分子。

【输入信息】
- 靶点 Uniprot ID: {info['uniprotid']}
- 靶点 ChEMBL ID: {info['chemblid']}
- 种子配体 SMILES（环数={info['seed_rc']} 个环）: {info['seed_smiles']}
{reference}
【目标属性】
- 属性名称: Ring Count（环数）
- 量化目标: 生成的每个分子的环数应为 {info['target_rc']} 个环（恰好比种子配体少一个环），且至少保留一个环。

【输出格式要求】
## Structural Analysis
（用1-2句话总结结构改造策略）

## Generated Molecules
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>
仅输出3个带标签的候选，不要在解释部分使用 smiles 标签。
这些是设计假设，尚未实际对接；不要编造 QED、SA、pKi 或 Vina 数值。
'''

def optimize_turn(info):
    return f'''依据前文靶点、文献、种子以及程序计算的候选结果，进行一次分子优化。
硬条件：Ring Count 恰好为 {info['target_rc']}，即种子环数 {info['seed_rc']} 减一。
仅设计单个连通分子，不要输出种子或输入的参考分子。保留合理的药效团，
在满足硬条件下提升类药性并注意合成可行性；不要为了结合力增加环数。
程序会将本轮3个候选与上一轮3个候选合并，按实际 QED 从高到低排序；
QED 完全相同时按实际 SA 原始分数从低到高排序，再按 canonical SMILES 排序。
未进行对接，不要声称运行 RDKit、QSAR、Vina 或虚构性质数值。
简短说明优化策略，然后只输出3个候选：
<smiles>SMILES1</smiles>
<smiles>SMILES2</smiles>
<smiles>SMILES3</smiles>'''

def main():
    if (OUT / 'locked.json').exists():
        raise SystemExit('Inputs already locked; refuse to overwrite')
    for name in ('inputs', 'logs', 'requests', 'responses', 'results', 'analysis', 'docking'):
        (OUT / name).mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    RDLogger.DisableLog('rdApp.*')
    model_path = EXP / 'models/best_gin_rc.pth'
    library_path = EXP / 'gin_week_20261008/data/library.csv'
    prompts_path = EXP / 'prompts/exp30rc25_pg_mild.jsonl'
    manifest_path = EXP / 'docking/receptor_manifest.json'
    lib = pd.read_csv(library_path)
    assert len(lib) == 26108 and lib.canonical.is_unique
    # Library normalization is fixed, not an opportunity to choose new favorable molecules.
    assert all(canon(smi) == smi for smi in lib.canonical)
    shutil.copy2(library_path, OUT/'inputs/library.csv')
    shutil.copy2(model_path, OUT/'inputs/best_gin_rc.pth')
    dump(OUT/'inputs/encoder_source.json',dict(GINEncoder=inspect.getsource(GINEncoder),
        mol_to_graph=inspect.getsource(mol_to_graph),original_run_sha256=sha(PROJECT/'run.py')))
    rc = lib.RingCount.to_numpy(int)
    keys = lib.canonical.to_numpy(str)
    fps = [FPGEN.GetFingerprint(Chem.MolFromSmiles(smi)) for smi in lib.canonical]
    counts = np.array([x.GetNumOnBits() for x in fps], dtype=float)
    saved = read_jsonl(prompts_path)
    targets = [r['chemblid'] for r in saved]
    assert len(targets) == len(set(targets)) == 25
    prompt_map = {r['chemblid']: r for r in saved}
    assert len(prompt_map) == len(saved) == 25 and set(prompt_map) == set(targets)
    graphs = [mol_to_graph(smi) for smi in lib.canonical]
    assert all(g is not None for g in graphs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GINEncoder(in_dim=graphs[0].x.size(1), hidden_dim=256, out_dim=256, num_layers=5).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True), strict=True)
    model.eval()
    embeddings = embed(model, graphs, device)
    repeat_a = embed(model, graphs[:32], device)
    repeat_b = embed(model, graphs[:32], device)
    assert np.array_equal(repeat_a, repeat_b)
    np.save(OUT / 'inputs/old_GIN_library_embeddings.npy', embeddings)
    target_records, example_rows, prompt_rows, checks = [], [], [], []
    for target in targets:
        turns = prompt_map[target]['prompt']
        assert len(turns) == 6 and all(isinstance(x, str) for x in turns)
        seed = re.search(r'- 种子配体 SMILES[^\n]*?: ([^\n]+)', turns[3]).group(1).strip()
        uniprot = re.search(r'- 靶点 Uniprot ID: ([^\n]+)', turns[3]).group(1).strip()
        seed_props = properties(seed)
        assert seed_props['RC'] >= 2
        info = dict(chemblid=target, uniprotid=uniprot, seed_smiles=seed,
                    seed_canonical=seed_props['canonical'], seed_rc=seed_props['RC'],
                    target_rc=seed_props['RC'] - 1, **{k: seed_props[k] for k in ('QED','SA_raw','SA_norm','MW','HeavyAtoms')})
        target_records.append(info)
        eligible = (rc == info['target_rc']) & (keys != info['seed_canonical'])
        qfp = FPGEN.GetFingerprint(Chem.MolFromSmiles(seed))
        intersection = np.array(DataStructs.BulkTanimotoSimilarity(qfp, fps))
        # Reconstruct binary cosine from Tanimoto and popcounts, not rank by Tanimoto.
        shared = np.rint(intersection * (qfp.GetNumOnBits() + counts) / (1. + intersection))
        m_scores = shared / np.sqrt(np.maximum(qfp.GetNumOnBits() * counts, 1e-10))
        qgraph = mol_to_graph(seed)
        q_a, q_b = embed(model, [qgraph], device), embed(model, [qgraph], device)
        assert np.array_equal(q_a, q_b)
        g_scores = embeddings @ q_a[0]
        choices = {'M': topk(m_scores, eligible, keys), 'G': topk(g_scores, eligible, keys)}
        for group in ('T', 'M', 'G'):
            example_smiles = None if group == 'T' else lib.iloc[choices[group]].canonical.tolist()
            final_turns = list(turns[:5])
            final_turns[0] = ('你是分子设计助手。正确使用 SMILES 的元素、芳香性、价态、环闭合及立体化学。'
                '环数按 RDKit RingCount 定义，不是 SMILES 数字的个数。后续只提供设计假设；'
                '实际化学校验、QED、SA 和对接由外部程序完成，不得声称自行运行了这些工具。')
            context_note = (f'\n靶点 ChEMBL ID: {target}；UniProt ID: {uniprot}。'
                '只总结输入确实支持的信息。摘要不一定包含小分子结合实验；结合位点为空时须明确说明未知，'
                '不要推断不存在于输入的口袋坐标、已知活性或对接分数。')
            final_turns[1] += context_note
            final_turns[2] += context_note
            final_turns[3] = property_turn(info, example_smiles)
            final_turns[4] = optimize_turn(info)
            prompt_rows.append(dict(chemblid=target, group=group, prompt=final_turns,
                                    exemplars=example_smiles or [], seed_smiles=seed,
                                    seed_rc=info['seed_rc'], target_rc=info['target_rc']))
            if example_smiles:
                assert len(set(example_smiles)) == 3
                for rank, (idx, smi) in enumerate(zip(choices[group], example_smiles), 1):
                    record = properties(smi)
                    assert record['RC'] == info['target_rc'] and record['canonical'] != info['seed_canonical']
                    example_rows.append(dict(chemblid=target, group=group, rank=rank,
                        library_id=lib.iloc[idx].query_id, smiles=smi,
                        similarity=float((m_scores if group == 'M' else g_scores)[idx]),
                        tanimoto=float(intersection[idx]), **record))
        mg = [p for p in prompt_rows if p['chemblid'] == target and p['group'] in ('M', 'G')]
        masked = [re.sub(r'Exemplar \d+（环数=\d+）: [^\n]+', 'EXEMPLAR', x['prompt'][3]) for x in mg]
        assert masked[0] == masked[1]
        assert all(mg[0]['prompt'][i] == mg[1]['prompt'][i] == prompt_rows[-3]['prompt'][i] for i in (0,1,2,4))
        checks.append(dict(chemblid=target, n_eligible=int(eligible.sum()),
                           m_g_shared=len(set(mg[0]['exemplars']) & set(mg[1]['exemplars'])), repeat_query_maxdiff=0.))
    pd.DataFrame(target_records).to_csv(OUT / 'inputs/targets.csv', index=False)
    pd.DataFrame(example_rows).to_csv(OUT / 'inputs/exemplars.csv', index=False)
    with (OUT / 'inputs/prompts.jsonl').open('w', encoding='utf-8') as stream:
        for row in prompt_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    # Interleaved target-position blocks, with fixed within-block group order.
    rng = np.random.default_rng(42)
    slots = []
    for position in range(1, 6):
        for target in targets:
            for group in rng.permutation(['T', 'M', 'G']):
                slots.append(dict(slot_id=f'{target}_{position}_{group}', chemblid=target,
                                  position=position, group=str(group), order=len(slots)+1))
    dump(OUT / 'inputs/schedule.json', slots)
    dump(OUT / 'inputs/precheck.json', dict(passed=True, device=str(device), targets=25,
         examples=150, prompts=75, slots=375, strict_determinism=True,
         library_repeat_maxdiff=0., per_target=checks))
    decisions_path = OUT / 'receptor_audit/decisions.json'
    decisions = json.loads(decisions_path.read_text())
    assert set(decisions) == set(targets), 'Complete receptor review before locking generation'
    receptors = json.loads(manifest_path.read_text())
    selected_receptors = {t: decisions[t]['receptor'] for t in targets if decisions[t]['include_docking']}
    dump(OUT / 'inputs/receptors.json', selected_receptors)
    sa, sa_provenance = scorer()
    import deepseek
    from openai import OpenAI
    import requests
    base_url = deepseek._get_base_url().rstrip('/')
    assert base_url in ('https://api.deepseek.com', 'https://api.deepseek.com/v1')
    client = OpenAI(api_key=deepseek._get_api_key(), base_url=base_url, max_retries=0, timeout=30)
    available = [x.id for x in client.models.list().data]
    assert 'deepseek-flash' in available
    response = requests.get('https://api.deepseek.com/user/balance',
                            headers={'Authorization':'Bearer '+deepseek._get_api_key()}, timeout=30)
    response.raise_for_status()
    balance = response.json()
    cny = [x for x in balance['balance_infos'] if x['currency'] == 'CNY']
    assert balance['is_available'] and len(cny) == 1
    total = float(cny[0]['total_balance'])
    assert total >= 75., 'Balance must cover CNY70 cap plus CNY5 reserve; never auto-recharge'
    dump(OUT / 'inputs/api_probe.json', dict(time_utc=now(), base_url=base_url,
         available_models=available, balance_before_cny=total, key_recorded=False))
    protocol = dict(created_utc=now(), status='locked_before_generation', groups=['T','M','G'],
        targets=targets, n_per_target=5, planned_slots=375, conversation_turns=5,
        target='exact RC_seed - 1 >= 1', shared_candidate_library_size=26108,
        selection_order='common exact-RC eligible set, then representation cosine top3',
        no_quality_reference_prefilter=True, no_tanimoto_hard_gate=True,
        checkpoint=str(model_path), checkpoint_sha256=sha(model_path),
        pretrain_finetune_causal_claim=False, historical_targets_not_new_external_test=True,
        system_prompt='你是分子设计助手。使用输入证据提出设计假设，不得虚构已执行工具或实验结果。',
        api=dict(model='deepseek-flash', documented_version='DeepSeek-V4.1-Flash',
                 base_url=base_url, thinking='enabled', reasoning_effort='low',
                 temperature=None, top_p=None, max_tokens=16384, timeout_seconds=300,
                 sdk_max_retries=0, max_whole_attempts_per_slot=2, workers=12,
                 cost_cap_cny=70., balance_reserve_cny=5.,
                 conservative_peak_budget_rates_per_million=dict(input=2.,output=8.),
                 cny_rates_per_million=dict(off_peak=dict(cache_hit=.02,cache_miss=1.,output=4.),
                                             peak=dict(cache_hit=.04,cache_miss=2.,output=8.)),
                 pricing_source='https://api-docs.deepseek.com/zh-cn/quick_start/pricing/',
                 docs_source='https://api-docs.deepseek.com/guides/thinking_mode/',
                 unknown_usage_charged_at_reserved_upper_bound=True),
        extraction='first three SMILES tags in each of turns4/5; invalid/missing proposals consume quota; extra tags ignored and counted',
        candidate_quota=dict(design_turns=[4,5],proposals_per_turn=3,max_total=6),
        final_selection='eligible single-fragment exact-RC noncopy candidates from turns4/5; QED descending, SA_raw ascending on exact QED ties, canonical SMILES ascending',
        program_feedback='actual RDKit/SA properties of fixed turn4 candidates appended to turn5; no docking feedback',
        retry='only transport/empty/truncated failures; no valid/eligible candidate is scientific failure and is never retried',
        primary=dict(comparison='G-M', metrics=['exact_noncopy_success_rate','compliant_QED_yield_per_planned_slot'],
                     multiplicity='Holm over two primary G-M endpoints',
                     unit='target-level paired mean', bootstrap_n=10000, bootstrap_seed=42,
                     permutation='100000 Monte Carlo target-wise sign flips, seed42; double-sided plus-one correction; exact enumeration if nonmissing targets<=15'),
        secondary_metrics=['SA_raw_mean','SA_norm_mean','vina_mean','dVina_mean','valid_rate',
                           'copy_rate','duplicate_rate','tanimoto_mean','QED_mean','MW_mean'],
        secondary_multiplicity='Holm over the two primary endpoints for G-T and M-T (four tests); other properties descriptive with unadjusted intervals and no significance claims',
        quality_subgroup='target-compliant QED/SA distributions are descriptive, post-treatment subset',
        failure_denominator='five planned slots per target; failed actual properties missing; explicit compliant-QED delivery utility zero for no output',
        docking=dict(available_targets=list(selected_receptors), excluded_targets=[t for t in targets if t not in selected_receptors],
                     subset_rule='pre-generation receptor identity/pocket audit, never output scores; exploratory computational docking only',
                     software='Vina 1.2.7', ligand_prep='RDKit Embed seed42 fallback random coords; MMFF500; Meeko; largest fragment',
                     sf_name='vina', exhaustiveness=16, n_poses=10, cpu_per_job=1,
                     seed=42, workers=8, timeout_seconds=600,
                     primary_score='best docked pose affinity: Vina.energies(n_poses=1)[0,0]',
                     seed_note='fixed42 rather than historical seed0 (automatic random); all arms/seeds rerun',
                     positive_scores_retained=True, do_not_filter_by_generated_quality=True),
        software={name:importlib.metadata.version(name) for name in ('torch','torch-geometric','rdkit','openai','vina','meeko')},
        sa_scorer=sa_provenance, no_manuscript_edits=True)
    dump(OUT / 'protocol.json', protocol)
    protected = json.loads((EXP / 'gin_budget_20261010/inputs/protected_hashes_before.json').read_text())
    extra = [model_path, PROJECT/'run.py', PROJECT/'deepseek.py', manifest_path,
             prompts_path] + [EXP / r['receptor_pdbqt'] for r in selected_receptors.values()]
    for path in extra:
        protected[str(path)] = sha(path)
    assert all(sha(path) == expected for path,expected in protected.items())
    dump(OUT / 'inputs/protected_hashes_before.json', protected)
    files = ['protocol.json','common.py','prepare_inputs.py','run_generation.py','run_docking.py','analyze_results.py',
             'inputs/targets.csv','inputs/exemplars.csv','inputs/prompts.jsonl','inputs/schedule.json',
             'inputs/precheck.json','inputs/receptors.json','inputs/protected_hashes_before.json','inputs/api_probe.json',
             'receptor_audit/decisions.json','AUTHORIZATION.md','verify_results.py',
             'inputs/library.csv','inputs/best_gin_rc.pth','inputs/encoder_source.json','README.md']
    dump(OUT / 'locked.json', dict(time_utc=now(), files={f:sha(OUT/f) for f in files},
         readonly_inputs={str(p):sha(p) for p in extra+[library_path]},
         rule='Do not change scientific protocol after generation; any technical analysis patch documented separately'))
    print('PRECHECK PASSED; LOCKED 375 SLOTS. NO PAID GENERATION YET.', flush=True)

if __name__ == '__main__':
    main()
