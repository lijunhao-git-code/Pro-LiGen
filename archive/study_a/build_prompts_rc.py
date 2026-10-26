"""
build_prompts_rc.py — Ring_Count 版 TG / PG-aggr prompt 构建（新文件，不改 build_prompts.py）

与 MW 版协议对齐，差异：
- 靶点名单固定为 exp30v2_pg_mw.jsonl 的同一批 25 个 chemblid（不重新抽样，保证 v2/v3 一致性）
- TG：Turn 4 追加环数约束文本（TG_SUFFIX_RC）
- PG-aggr：Turn 3 替换为环数 exemplar 模板（目标：环数 ≤ max(1, min exemplar 环数) 且 ≥1 环）
- NG 不需要生成：NG 与目标属性无关，直接复用 MW 实验的 exp30v2_ng 125 分子
- 只输出检索合格靶点（PG 与 TG 同口径）

用法：
    python3 mw_experiments/build_prompts_rc.py \
        --retrieval mw_experiments/retrieval/retrieved_rc_filtered.csv
"""
import argparse
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROMT_JSONL = BASE.parent / "prompt.jsonl"
PG25_JSONL = BASE / "prompts" / "exp30v2_pg_mw.jsonl"   # 25 靶点名单来源（与 v2/v3 同一批）


def load_rc_retrieval(retrieval_csv: str):
    """加载 RC 检索结果：chemblid -> info；跳过 status=failed"""
    import csv as csv_module
    mapping = {}
    n_skipped = 0
    with open(retrieval_csv, 'r', encoding='utf-8') as f:
        for row in csv_module.DictReader(f):
            chemblid = row.get('chemblid', '').strip()
            if not chemblid:
                continue
            if row.get('status', 'ok').strip().lower() == 'failed':
                n_skipped += 1
                continue
            mapping[chemblid] = {
                'chemblid': chemblid,
                'uniprotid': row.get('uniprotid', '').strip(),
                'best_ligand_smiles': row.get('best_ligand_smiles', '').strip(),
                'top1_smiles': row.get('top1_smiles', '').strip(),
                'top2_smiles': row.get('top2_smiles', '').strip(),
                'top3_smiles': row.get('top3_smiles', '').strip(),
                'seed_rc': _to_int(row.get('seed_rc')),
                'ex1_rc': _to_int(row.get('ex1_rc')),
                'ex2_rc': _to_int(row.get('ex2_rc')),
                'ex3_rc': _to_int(row.get('ex3_rc')),
            }
    print(f"加载 RC 检索结果: {len(mapping)} 个合格靶点（跳过 {n_skipped} 个检索失败靶点）")
    return mapping


def _to_int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def build_pg_aggr_prompt(info: dict) -> str:
    """PG-aggr 环数版属性导向 prompt（替换 Turn 3）"""
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
        ub = max(1, min(valid_ex))
        target_line = (
            f"- 量化目标: 生成的每个分子的环数不超过 {ub} 个环"
            f"（即不高于环数最少的 exemplar），且至少保留 1 个环"
            f"（避免完全开环破坏与靶点口袋的接触）。"
        )
    else:
        target_line = "- 量化目标: 生成的每个分子的环数应明显少于种子配体，但至少保留 1 个环。"

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


TG_SUFFIX_RC = """

【额外约束 - 环数优化】
请特别注意：生成的分子应具有较少的环数（Ring Count），即分子骨架中闭环结构的数量应少于参考配体。
较少的环通常意味着更简单的分子拓扑和更高的合成可及性。
请在分子设计时优先考虑：
1. 移除对靶点结合非必需的芳香环或脂肪环
2. 避免构建稠环、螺环、桥环等复杂环系
3. 用开链或简单片段替代非关键环（但保留药效团所需的关键环）
"""


def main():
    parser = argparse.ArgumentParser(description='构建 RC 版 TG / PG-aggr prompt（新文件）')
    parser.add_argument('--retrieval', type=str,
                        default=str(BASE / 'retrieval' / 'retrieved_rc_filtered.csv'))
    parser.add_argument('--outdir', type=str, default=str(BASE / 'prompts'))
    args = parser.parse_args()

    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 固定 25 靶点名单（与 v2/v3 同一批，不重新抽样）
    target25 = [json.loads(l)['chemblid'] for l in open(PG25_JSONL, encoding='utf-8') if l.strip()]
    print(f"固定靶点名单（v2/v3 同批）: {len(target25)} 个")

    rmap = load_rc_retrieval(args.retrieval)
    ok_targets = sorted(set(target25) & set(rmap.keys()))
    print(f"25 靶点中 RC 检索合格: {len(ok_targets)} 个")

    # 读取 prompt.jsonl 的原始条目
    entries = {}
    with open(PROMT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            entries[e.get('chemblid', '')] = e

    tg_path = out_dir / 'exp30rc_tg.jsonl'
    pg_path = out_dir / 'exp30rc_pg_aggr.jsonl'
    tg_count = pg_count = 0

    with open(tg_path, 'w', encoding='utf-8') as ftg, \
         open(pg_path, 'w', encoding='utf-8') as fpg:
        for chemblid in ok_targets:
            entry = entries.get(chemblid)
            if entry is None:
                print(f"警告: {chemblid} 不在 prompt.jsonl 中，跳过")
                continue
            prompt_list = entry.get('prompt', [])
            if len(prompt_list) < 6:
                print(f"警告: {chemblid} 的 prompt 不足 6 轮，跳过")
                continue

            # ---- TG: 删除 Turn 3，Turn 4（删除后索引 3）追加环数约束 ----
            tg_prompt = [p for i, p in enumerate(prompt_list) if i != 3]
            if len(tg_prompt) > 3:
                tg_prompt[3] = tg_prompt[3] + TG_SUFFIX_RC
            ftg.write(json.dumps(
                {'chemblid': chemblid, 'prompt': tg_prompt, 'variant': 'TG'},
                ensure_ascii=False) + '\n')
            tg_count += 1

            # ---- PG-aggr: Turn 3 替换为环数 exemplar 模板 ----
            info = rmap[chemblid]
            new_turn3 = build_pg_aggr_prompt(info)
            pg_prompt = list(prompt_list)
            pg_prompt[3] = new_turn3
            fpg.write(json.dumps(
                {'chemblid': chemblid, 'prompt': pg_prompt, 'variant': 'PG-aggr'},
                ensure_ascii=False) + '\n')
            pg_count += 1

    print(f"\n生成结果:")
    print(f"  TG:      {tg_count} 条 -> {tg_path}")
    print(f"  PG-aggr: {pg_count} 条 -> {pg_path}")
    print(f"  NG:      复用 MW 实验 exp30v2_ng 的 125 分子（属性无关，不重新生成）")


if __name__ == '__main__':
    main()
