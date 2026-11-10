"""
select_expand_targets_rc.py — 为 25×10 增强实验补选 6 个新靶点（新文件）

选择规则（确定性、可复现）：
  - 候选：retrieved_rc_k60.csv 中 status=ok（RC 检索合格）
  - 排除：种子环数 <2（环数目标无意义）、seed MW > 700（巨型肽）、CHEMBL3401（docking 无代表性口袋）
  - 排除：原 25 靶点名单（exp30v2_pg_mw.jsonl）
  - random.seed(42) 抽 6 个

输出：
  - prompts/rc_expand_new6.jsonl     ：新 6 靶点的原始 prompt.jsonl 条目（供构建 NG/TG/PG prompt）
  - retrieval/rc_expand_targets.csv ：新 6 靶点名单 + 种子信息（供复核）
"""
import csv
import json
import random
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

RDLogger.DisableLog('rdApp.*')

BASE = Path(__file__).resolve().parent
RC_RETR = BASE / "retrieval" / "retrieved_rc_k60.csv"
PG25 = BASE / "prompts" / "exp30v2_pg_mw.jsonl"
PROMPT_JSONL = BASE.parent / "prompt.jsonl"
OUT_JSONL = BASE / "prompts" / "rc_expand_new6.jsonl"
OUT_CSV = BASE / "retrieval" / "rc_expand_targets.csv"


def ring_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.RingCount(mol) if mol is not None else None


def mw(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.MolWt(mol) if mol is not None else None


def main():
    # 原 25 靶点名单
    old25 = [json.loads(l)["chemblid"] for l in open(PG25, encoding="utf-8") if l.strip()]
    old_set = set(old25)
    print(f"原 25 靶点名单: {len(old_set)}")

    # 候选：RC 检索合格
    cands = []
    for r in csv.DictReader(open(RC_RETR, encoding="utf-8")):
        if r.get("status") != "ok":
            continue
        c = r["chemblid"]
        if c in old_set or c == "CHEMBL3401":
            continue
        rc = ring_count(r["best_ligand_smiles"])
        m = mw(r["best_ligand_smiles"])
        if rc is None or rc < 2:
            continue
        if m is None or m > 700:
            continue
        cands.append({"chemblid": c, "seed_rc": int(rc), "seed_mw": round(m, 1),
                      "best_ligand_smiles": r["best_ligand_smiles"]})
    print(f"合格候选靶点: {len(cands)}")

    random.seed(42)
    picked = random.sample(cands, 6)
    print(f"补选 6 个新靶点:")
    for p in sorted(picked, key=lambda x: x["chemblid"]):
        print(f"  {p['chemblid']}: seed {p['seed_rc']} 环, MW {p['seed_mw']}")

    # 输出名单 CSV
    import pandas as pd
    pd.DataFrame(picked).to_csv(OUT_CSV, index=False)
    print(f"[输出] {OUT_CSV}")

    # 输出新 6 靶点的原始 prompt 条目
    entries = {}
    with open(PROMPT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            entries[e.get("chemblid", "")] = e

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for p in sorted(picked, key=lambda x: x["chemblid"]):
            c = p["chemblid"]
            if c not in entries:
                print(f"警告: {c} 不在 prompt.jsonl 中，跳过")
                continue
            f.write(json.dumps(entries[c], ensure_ascii=False) + "\n")
    print(f"[输出] {OUT_JSONL}")


if __name__ == "__main__":
    main()
