"""
generate_parallel.py — 多线程并行 LLM 分子生成（ThreadPoolExecutor）

用法：
    # 基础用法
    python mw_experiments/generate_parallel.py \
        --input mw_experiments/prompts/exp30_ng.jsonl \
        --output mw_experiments/generated/exp30_ng.txt \
        -n 5 -w 10

    # 断点续跑
    python mw_experiments/generate_parallel.py \
        --input mw_experiments/prompts/exp30_ng.jsonl \
        --output mw_experiments/generated/exp30_ng.txt \
        -n 5 -w 10 --resume

设计要点：
- ThreadPoolExecutor（API 调用是 I/O 密集型，线程足够）
- 每个 worker 独立创建 DeepSeek client（线程安全）
- 随机启动延迟避免同时发请求被限流
- 支持 checkpoint/resume（跳过已生成的靶点-分子组合）
- 实时进度 + 失败率统计
"""

import json
import re
import argparse
import time
import random
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from rdkit import Chem
import sys

# 将父目录加入 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepseek import chat as deepseek_chat

# ================== 常量 ==================
MAX_RETRIES = 10
RETRY_DELAY = 2.0         # 重试间隔 (秒)
STARTUP_JITTER = 3.0      # 启动随机延迟上限 (秒)
API_CALL_GAP = 0.5        # 同一 worker 连续请求间隔 (秒)

# ================== 工具函数 ==================
def is_valid_smiles(smiles: str) -> bool:
    if not smiles or not isinstance(smiles, str):
        return False
    mol = Chem.MolFromSmiles(smiles.strip())
    return mol is not None


def extract_smiles(text: str) -> str | None:
    """从 LLM 回复中提取 SMILES"""
    # 优先 <SMILES>...</SMILES>：遍历所有标签匹配，返回第一个 RDKit 合法的
    for tag in ['<SMILES>', '<smiles>']:
        end_tag = tag.replace('<', '</')
        matches = re.findall(
            re.escape(tag) + r'\s*(.*?)\s*' + re.escape(end_tag),
            text, re.DOTALL | re.IGNORECASE
        )
        for m in matches:
            candidate = m.strip()
            if candidate and is_valid_smiles(candidate):
                return candidate

    # 备选：长串 SMILES 模式（逐行校验合法性）
    smiles_pattern = re.findall(
        r'(?:^|\n)\s*([A-Za-z0-9@\[\]\(\)/\\#%=.+:\-]{10,})\s*(?:\n|$)',
        text, re.MULTILINE
    )
    for candidate in smiles_pattern:
        candidate = candidate.strip()
        if is_valid_smiles(candidate):
            return candidate
    return None


# ================== 单分子生成任务 ==================
def generate_one(prompt_list: list, chemblid: str, gen_idx: int) -> dict:
    """
    生成一个分子。每个线程独立调用（deepseek.chat 内部创建独立 client）。
    返回: {'smiles': str|None, 'chemblid': str, 'gen_idx': int, 'ok': bool}
    """
    for attempt in range(MAX_RETRIES):
        try:
            answer = deepseek_chat(prompt_list)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            continue

        smiles = extract_smiles(answer)
        if smiles and is_valid_smiles(smiles):
            return {'smiles': smiles, 'chemblid': chemblid, 'gen_idx': gen_idx, 'ok': True}

        time.sleep(RETRY_DELAY)

    return {'smiles': None, 'chemblid': chemblid, 'gen_idx': gen_idx, 'ok': False}


# ================== 单个靶点的全部分子生成 ==================
def process_one_target(entry: dict, n_per_target: int, worker_id: int,
                       completed_set: set) -> list[dict]:
    """
    处理一个靶点：生成 n_per_target 个分子。
    completed_set: 已完成的 (chemblid, gen_idx) 集合，用于断点续跑。
    """
    # 随机延迟避免所有线程同时发起第一个请求
    time.sleep(random.uniform(0, STARTUP_JITTER))

    chemblid = entry.get('chemblid', 'UNKNOWN')
    prompt_list = entry.get('prompt', [])
    variant = entry.get('variant', 'UNKNOWN')

    if not prompt_list:
        return []

    results = []
    for gen_idx in range(1, n_per_target + 1):
        # 断点续跑：跳过已完成
        if (chemblid, gen_idx) in completed_set:
            continue

        result = generate_one(prompt_list, chemblid, gen_idx)
        result['variant'] = variant
        results.append(result)

        status = 'OK' if result['ok'] else 'FAIL'
        smi_preview = (result['smiles'][:55] + '...') if result['smiles'] else '(invalid)'
        print(f"[W{worker_id:02d}] {chemblid} [{gen_idx}/{n_per_target}] {status}  {smi_preview}")

        # 请求间隔（避免触发速率限制）
        time.sleep(API_CALL_GAP)

    n_ok = sum(1 for r in results if r['ok'])
    n_total = len(results)
    print(f"[W{worker_id:02d}] DONE {chemblid} ({variant}): {n_ok}/{n_total} valid")
    return results


# ================== 加载断点 ==================
def load_checkpoint(output_path: str) -> set:
    """从已有输出文件加载已完成的 (chemblid, gen_idx) 集合，用于断点续跑"""
    completed = set()
    path = Path(output_path)
    if not path.exists():
        return completed

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 格式: "SMILES || chemblid || variant || gen_N"
            parts = line.split(' || ')
            if len(parts) >= 4:
                chemblid = parts[1].strip()
                gen_str = parts[3].strip()
                try:
                    gen_idx = int(gen_str.replace('gen_', ''))
                except ValueError:
                    continue
                completed.add((chemblid, gen_idx))
    return completed


# ================== 主逻辑 ==================
def main():
    parser = argparse.ArgumentParser(description='并行 LLM 分子生成 (ThreadPoolExecutor)')
    parser.add_argument('--input', type=str, required=True, help='输入 prompt JSONL')
    parser.add_argument('--output', type=str, required=True, help='输出文件')
    parser.add_argument('--n_per_target', '-n', type=int, default=5,
                        help='每个靶点生成分子数 (默认 5)')
    parser.add_argument('--workers', '-w', type=int, default=10,
                        help='并行线程数 (默认 10)')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='断点续跑（跳过已生成的分子）')
    parser.add_argument('--jitter', type=float, default=3.0,
                        help='启动随机延迟上限秒数 (默认 3.0)')
    args = parser.parse_args()

    # 覆盖全局 jitter
    global STARTUP_JITTER
    STARTUP_JITTER = args.jitter

    # ===== 读取 prompts =====
    entries = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))

    n_targets = len(entries)
    total_jobs = n_targets * args.n_per_target

    print(f"{'='*55}")
    print(f"Input:   {args.input}")
    print(f"Targets: {n_targets}")
    print(f"Mol/target: {args.n_per_target}")
    print(f"Total:   {total_jobs} molecules")
    print(f"Workers: {args.workers} threads")
    print(f"Output:  {args.output}")
    print(f"Resume:  {'YES' if args.resume else 'NO'}")
    print(f"Jitter:  {args.jitter}s")
    print(f"{'='*55}")

    # ===== 断点续跑 =====
    completed_set = set()
    if args.resume:
        completed_set = load_checkpoint(args.output)
        remaining = total_jobs - len(completed_set)
        print(f"Checkpoint: {len(completed_set)}/{total_jobs} done, {remaining} remaining")
        if remaining <= 0:
            print("All molecules already generated. Nothing to do.")
            return

    # ===== 确保输出目录存在 =====
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ===== 线程安全写入 =====
    write_lock = threading.Lock()

    def safe_write(line: str):
        with write_lock:
            with open(args.output, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
                f.flush()

    # ===== 并行生成 =====
    t_start = time.time()
    total_ok = 0
    total_fail = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {}
        for i, entry in enumerate(entries):
            worker_id = i % args.workers  # 轮转分配
            future = executor.submit(
                process_one_target, entry, args.n_per_target,
                worker_id, completed_set
            )
            futures[future] = entry.get('chemblid', f'UNKNOWN_{i}')

        for future in as_completed(futures):
            chemblid = futures[future]
            try:
                results = future.result()
            except Exception as e:
                print(f"ERROR {chemblid}: {e}")
                continue

            for r in results:
                if r['ok']:
                    line = f"{r['smiles']} || {r['chemblid']} || {r['variant']} || gen_{r['gen_idx']}"
                    safe_write(line)
                    total_ok += 1
                else:
                    total_fail += 1

    # ===== 统计 =====
    elapsed = time.time() - t_start
    total = total_ok + total_fail
    success_rate = 100 * total_ok / max(total, 1)

    print(f"\n{'='*55}")
    print(f"ALL DONE!")
    print(f"  Time:       {elapsed/60:.1f} min")
    print(f"  Successful: {total_ok}/{total} ({success_rate:.1f}%)")
    print(f"  Failed:     {total_fail}")
    print(f"  Output:     {args.output}")
    print(f"{'='*55}")


if __name__ == '__main__':
    main()
