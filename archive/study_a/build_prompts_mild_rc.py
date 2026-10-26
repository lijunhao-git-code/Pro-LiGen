"""
build_prompts_mild_rc.py — PG-mild (Ring_Count) 目标严格度消融 prompt 构建（新文件）

与 build_prompts_rc.py 的 PG-aggr 同靶点、同检索、同结构，唯一差异是量化目标：
- PG-aggr：环数 ≤ max(1, min exemplar 环数) 且 ≥1（激进）
- PG-mild：环数恰好 = seed − 1 环（温和降一档，保留骨架）

用法：
    python3 mw_experiments/build_prompts_mild_rc.py \
        --retrieval mw_experiments/retrieval/retrieved_rc_filtered.csv
"""
import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from build_prompts_rc import load_rc_retrieval, PG25_JSONL, PROMT_JSONL


def build_pg_mild_prompt(info: dict) -> str:
    """PG-mild 环数版属性导向 prompt（替换 Turn 3）：目标 = 种子环数 − 1"""
    seed_rc = info.get('seed_rc')
    ex_rcs = [info.get('ex1_rc'), info.get('ex2_rc'), info.get('ex3_rc')]
    valid_ex = [r for r in ex_rcs if r is not None]

    seed_txt = f"{seed_rc} 个环" if seed_rc is not None else "未知"
    ex_lines = []
    for i, (smi, rc) in enumerate(zip(
            [info['top1_smiles'], info['top2_smiles'], info['top3_smiles']],
            ex_rcs), start=1):
        rc_txt = f"（环数={rc}）" if rc is not None else ""
        ex_lines.append(f"  - Exemplar {i}{rc_txt}: {smi}")
    ex_block = "\n".join(ex_lines)

    if seed_rc is not None and valid_ex:
        target_rc = max(1, seed_rc - 1)
        target_line = (
            f"- 量化目标: 生成的每个分子的环数应为 {target_rc} 个环"
            f"（恰好比种子配体少一个环）。"
            f"即：温和地简化一个环系，同时保留其余关键骨架与口袋接触。"
        )
    else:
        target_line = "- 量化目标: 生成的每个分子的环数应比种子配体少一个环，但不要过度简化。"

    return f"""你是一位经验丰富的药物化学专家，擅长基于结构相似的示例分子进行启发式分子设计。

【任务】
给定一个目标靶点及其已知的结合配体（种子配体），同时提供三个与该种子配体结构相似、但环数（Ring Count）更少的示例分子（exemplars），并给出各自的环数。请你：
1. 分析这三个示例分子相比于种子配体，在结构上存在哪些共同的改造模式（例如：哪些环系被移除或打开、环系如何简化、稠环如何拆分、杂原子如何替换等）。
2. 基于这些改造模式，结合靶点信息，生成 **3个全新的、化学上有效的 SMILES 字符串**。这些新分子应保留与靶点结合的关键骨架，同时满足下方的量化目标。
3. 不要直接输出示例分子本身，也不要输出种子配体。

【输入信息】
- 靶点 Uniprot ID: {info['uniprotid']}
- 靶点 ChEMBL ID: {info['chemblid']}
- 种子配体 SMILES（与靶点结合最好的已知分子，环数={seed_txt}）: {info['best_ligand_smiles']}
- 结构相似且环数更少的示例分子（共3个）:
{ex_block}

【目标属性】
- 属性名称: Ring Count（环数）
- 优化方向: lower（更少的环 → 更简单的分子拓扑）
{target_line}

【输出格式要求】
请严格按照以下格式输出，不要添加额外解释：

## Structural Analysis of Exemplars
（用1-2句话总结三个示例分子相比于种子配体的共同结构改造策略）

## Generated Molecules
SMILES1
SMILES2
SMILES3
"""


def main():
    parser = argparse.ArgumentParser(description='构建 RC 版 PG-mild prompt（新文件）')
    parser.add_argument('--retrieval', type=str,
                        default=str(BASE / 'retrieval' / 'retrieved_rc_filtered.csv'))
    parser.add_argument('--outdir', type=str, default=str(BASE / 'prompts'))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target25 = [json.loads(l)['chemblid'] for l in open(PG25_JSONL, encoding='utf-8') if l.strip()]
    rmap = load_rc_retrieval(args.retrieval)
    ok_targets = sorted(set(target25) & set(rmap.keys()))
    print(f"25 靶点中 RC 检索合格: {len(ok_targets)} 个（PG-mild 与 PG-aggr 同靶点）")

    entries = {}
    with open(PROMT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            entries[e.get('chemblid', '')] = e

    out_path = out_dir / 'exp30rc_pg_mild.jsonl'
    n_out = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for chemblid in ok_targets:
            entry = entries.get(chemblid)
            if entry is None:
                continue
            prompt_list = entry.get('prompt', [])
            if len(prompt_list) < 6:
                continue
            pg_prompt = list(prompt_list)
            pg_prompt[3] = build_pg_mild_prompt(rmap[chemblid])
            f.write(json.dumps(
                {'chemblid': chemblid, 'prompt': pg_prompt, 'variant': 'PG-mild'},
                ensure_ascii=False) + '\n')
            n_out += 1

    print(f"PG-mild: {n_out} 条 -> {out_path}")


if __name__ == '__main__':
    main()
