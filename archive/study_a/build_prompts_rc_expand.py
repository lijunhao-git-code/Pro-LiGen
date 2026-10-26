"""
build_prompts_rc_expand.py — 25×10 增强实验的 prompt 构建（新文件，不改任何旧脚本）

组装 25 靶点（RC 合格 19 个 + 新选 6 个）的四变体 prompt：
  - NG     ：旧 exp30v2_ng_25.jsonl 中 RC19 靶点条目 + 新 6 靶点条目（删 Turn 3）
  - TG     ：旧 exp30rc_tg.jsonl（19）+ 新 6 靶点条目（复用 TG_SUFFIX_RC）
  - PG-aggr：旧 exp30rc_pg_aggr.jsonl（19）+ 新 6 靶点条目（复用 build_pg_aggr_prompt）
  - PG-mild：旧 exp30rc_pg_mild.jsonl（19）+ 新 6 靶点条目（复用 build_pg_mild_prompt）

输出前缀 exp30rc25_（新文件，旧 prompt 文件不动）：
  prompts/exp30rc25_{ng,tg,pg_aggr,pg_mild}.jsonl

用法：
    python3 mw_experiments/build_prompts_rc_expand.py
"""
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from build_prompts_rc import load_rc_retrieval, build_pg_aggr_prompt, TG_SUFFIX_RC
from build_prompts_mild_rc import build_pg_mild_prompt

NEW6_JSONL = BASE / "prompts" / "rc_expand_new6.jsonl"        # 新 6 靶点原始条目
RC_RETR = BASE / "retrieval" / "retrieved_rc_k60.csv"          # RC 检索（25 靶点内 19 合格）
OLD_NG = BASE / "prompts" / "exp30v2_ng_25.jsonl"              # MW 版 NG（25 靶点）
OLD_TG = BASE / "prompts" / "exp30rc_tg.jsonl"                 # RC TG（19）
OLD_AGGR = BASE / "prompts" / "exp30rc_pg_aggr.jsonl"          # RC PG-aggr（19）
OLD_MILD = BASE / "prompts" / "exp30rc_pg_mild.jsonl"          # RC PG-mild（19）
OUT_DIR = BASE / "prompts"


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_ng_entry(entry):
    """NG：删除 Turn 3（属性导向轮），与 build_prompts.py 逻辑一致"""
    prompt_list = entry.get("prompt", [])
    if len(prompt_list) < 6:
        return None
    return {"chemblid": entry["chemblid"],
            "prompt": [p for i, p in enumerate(prompt_list) if i != 3],
            "variant": "NG"}


def build_tg_entry(entry):
    """TG：删除 Turn 3，Turn 4（删除后索引 3）追加环数约束"""
    prompt_list = entry.get("prompt", [])
    if len(prompt_list) < 6:
        return None
    tg = [p for i, p in enumerate(prompt_list) if i != 3]
    if len(tg) > 3:
        tg[3] = tg[3] + TG_SUFFIX_RC
    return {"chemblid": entry["chemblid"], "prompt": tg, "variant": "TG"}


def build_pg_entry(entry, rmap, mild=False):
    """PG-aggr / PG-mild：Turn 3 替换为环数 exemplar 模板"""
    c = entry.get("chemblid", "")
    info = rmap.get(c)
    if info is None:
        return None
    prompt_list = entry.get("prompt", [])
    if len(prompt_list) < 6:
        return None
    turn3 = build_pg_mild_prompt(info) if mild else build_pg_aggr_prompt(info)
    pg = list(prompt_list)
    pg[3] = turn3
    return {"chemblid": c, "prompt": pg,
            "variant": "PG-mild" if mild else "PG-aggr"}


def main():
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    old_ng = load_jsonl(OLD_NG)
    old_tg = load_jsonl(OLD_TG)
    old_aggr = load_jsonl(OLD_AGGR)
    old_mild = load_jsonl(OLD_MILD)
    new6 = load_jsonl(NEW6_JSONL)

    # 旧 19 靶点名单（RC 合格 ∩ 原 25）
    rmap = load_rc_retrieval(str(RC_RETR))
    rc19 = {e["chemblid"] for e in load_jsonl(OLD_AGGR)}
    print(f"旧 19 靶点: {len(rc19)} | 新 6 靶点: {len(new6)}")

    # ---- NG：旧 19 靶点条目（来自 MW 版 NG）+ 新 6 条目 ----
    ng_entries = [e for e in old_ng if e["chemblid"] in rc19]
    # ---- TG/PG：旧 19 靶点条目 + 新 6 条目 ----
    tg_entries = list(old_tg)
    aggr_entries = list(old_aggr)
    mild_entries = list(old_mild)

    for e in new6:
        ng = build_ng_entry(e)
        tg = build_tg_entry(e)
        aggr = build_pg_entry(e, rmap, mild=False)
        mild = build_pg_entry(e, rmap, mild=True)
        if ng: ng_entries.append(ng)
        if tg: tg_entries.append(tg)
        if aggr: aggr_entries.append(aggr)
        if mild: mild_entries.append(mild)

    files = {
        "exp30rc25_ng.jsonl": ng_entries,
        "exp30rc25_tg.jsonl": tg_entries,
        "exp30rc25_pg_aggr.jsonl": aggr_entries,
        "exp30rc25_pg_mild.jsonl": mild_entries,
    }
    for name, entries in files.items():
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{name}: {len(entries)} 条 -> {path}")

    print("\n[下一步] python3 mw_experiments/prep_checkpoint_rc25.py  # 预载旧分子到新输出文件")


if __name__ == "__main__":
    main()
