"""
filter_prompts.py — 按 chemblid 集合过滤 prompt jsonl

用法：
    python3 mw_experiments/filter_prompts.py --input mw_experiments/prompts/exp30v2_ng.jsonl \
        --keep-ids-from mw_experiments/prompts/exp30v2_pg_mw.jsonl \
        --output mw_experiments/prompts/exp30v2_ng_25.jsonl
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--keep-ids-from", required=True, help="参考 jsonl，只保留其 chemblid 集合")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    keep = set()
    with open(args.keep_ids_from, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                keep.add(json.loads(line).get("chemblid"))

    n_out = 0
    with open(args.input, encoding="utf-8") as f, open(args.output, "w", encoding="utf-8") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            if e.get("chemblid") in keep:
                out.write(line + "\n")
                n_out += 1
    print(f"过滤完成: {n_out} 条 -> {args.output}")


if __name__ == "__main__":
    main()
