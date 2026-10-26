"""
retrieve_rc.py — 使用 Ring_Count-tuned GIN 检索结构相似但环数更少的 exemplar（新文件）

协议与 retrieve_mw.py 完全对齐（候选池 K=30 + 强制过滤 + 失败标记 + 嵌入缓存），
唯一区别是过滤条件：环数(seed) − 环数(candidate) ≥ min_rc_margin（默认 1，即至少少一个环）。
环数离散、值域小，0/1 环种子天然无法检索到"更少环"的 exemplar，会被标记 failed。

用法：
    python mw_experiments/retrieve_rc.py --model mw_experiments/models/best_gin_rc.pth \
        --candidate_k 30 --min_rc_margin 1 --output mw_experiments/retrieval/retrieved_rc_filtered.csv

输出列：uniprotid, chemblid, best_ligand_smiles, seed_rc, top1-3_smiles, ex1-3_rc, n_found, status
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# 复用 retrieve_mw.py 中与属性无关的工具函数（不改旧文件）
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from retrieve_mw import build_train_embeddings_cached, clean_smiles_column
from run import MolecularEmbedder


def ring_count(smiles):
    mol = Chem.MolFromSmiles(smiles)
    return Descriptors.RingCount(mol) if mol is not None else np.nan


def retrieve_rc_filtered(query_smiles, query_rc, embedder,
                         train_embeddings, train_smiles, train_rcs,
                         candidate_k=30, min_rc_margin=1, top_k=3):
    """在 top-candidate_k 相似候选内，强制筛选 环数(query) − 环数(cand) ≥ min_rc_margin 的候选，
    再按相似度降序取 top_k 个。返回 (smiles 列表, 环数列表, 找到数量)。
    """
    if not isinstance(query_smiles, str) or len(query_smiles.strip()) == 0:
        return [None] * top_k, [None] * top_k, 0
    emb = embedder.embed_single(query_smiles, return_numpy=True)
    if emb is None:
        return [None] * top_k, [None] * top_k, 0

    similarities = cosine_similarity(emb.reshape(1, -1), train_embeddings)[0]
    order = np.argsort(similarities)[::-1][:candidate_k]

    qmol = Chem.MolFromSmiles(query_smiles)
    query_canon = Chem.MolToSmiles(qmol) if qmol is not None else None

    cands = []
    seen = set()
    for idx in order:
        smi = train_smiles[idx]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen or (query_canon is not None and canon == query_canon):
            continue
        seen.add(canon)
        rc = train_rcs[idx]
        if (not np.isnan(rc)) and (not np.isnan(query_rc)) and (query_rc - rc) >= min_rc_margin:
            cands.append((similarities[idx], smi, rc))
        if len(cands) >= top_k:
            break

    n_found = len(cands)
    top_smiles = [c[1] for c in cands] + [None] * (top_k - n_found)
    top_rcs = [c[2] for c in cands] + [None] * (top_k - n_found)
    return top_smiles, top_rcs, n_found


def main():
    parser = argparse.ArgumentParser(description='检索 Ring_Count 优化的 exemplar（新文件）')
    parser.add_argument('--model', type=str, required=True, help='GIN 模型权重路径 (.pth)')
    parser.add_argument('--target_csv', type=str, default='csv_folder/best_ligand.csv')
    parser.add_argument('--train_csv', type=str, default='csv_folder/train.csv')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--candidate_k', type=int, default=30, help='候选池大小')
    parser.add_argument('--min_rc_margin', type=int, default=1, help='环数(seed)-环数(ex) 最小差值')
    parser.add_argument('--error_csv', type=str, default=None)
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    if args.output is None:
        args.output = str(base_dir / 'retrieval' / 'retrieved_rc_filtered.csv')
    if args.error_csv is None:
        args.error_csv = str(Path(args.output).with_suffix('.errors.csv'))

    print(f"Model: {args.model}")
    print(f"Output: {args.output}")
    print(f"候选池 K={args.candidate_k}, 环数阈值差={args.min_rc_margin}, 取 top-{args.k}")

    embedder = MolecularEmbedder(args.model)

    cache_dir = base_dir / 'retrieval_cache'
    train_embeddings, train_smiles = build_train_embeddings_cached(
        args.train_csv, embedder, cache_dir, model_tag=Path(args.model).stem)

    print("计算训练集环数 ...")
    train_rcs = np.array([ring_count(s) for s in tqdm(train_smiles, desc="RC")], dtype=float)

    df_target = pd.read_csv(args.target_csv)
    for col in ['uniprotid', 'chemblid', 'best_ligand_smiles']:
        if col not in df_target.columns:
            raise ValueError(f"目标 CSV 缺少列 '{col}'")

    df_target_clean, removed_target = clean_smiles_column(df_target, 'best_ligand_smiles')
    if removed_target:
        df_target.loc[removed_target].to_csv(args.error_csv, index=False)
        print(f"{len(removed_target)} 个无效查询 SMILES -> {args.error_csv}")
    if len(df_target_clean) == 0:
        print("无有效查询 SMILES，退出。")
        return

    df_target_clean['seed_rc'] = df_target_clean['best_ligand_smiles'].apply(ring_count)

    rows = []
    for _, row in tqdm(df_target_clean.iterrows(), total=len(df_target_clean), desc="检索+过滤"):
        query_smi = row['best_ligand_smiles']
        query_rc = row['seed_rc']
        top_smiles, top_rcs, n_found = retrieve_rc_filtered(
            query_smi, query_rc, embedder, train_embeddings, train_smiles, train_rcs,
            candidate_k=args.candidate_k, min_rc_margin=args.min_rc_margin, top_k=args.k)
        rows.append({
            'uniprotid': row['uniprotid'],
            'chemblid': row['chemblid'],
            'best_ligand_smiles': query_smi,
            'seed_rc': query_rc,
            'top1_smiles': top_smiles[0], 'top2_smiles': top_smiles[1], 'top3_smiles': top_smiles[2],
            'ex1_rc': top_rcs[0], 'ex2_rc': top_rcs[1], 'ex3_rc': top_rcs[2],
            'n_found': n_found,
            'status': 'ok' if n_found >= args.k else 'failed',
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(args.output, index=False)
    n_ok = (df_out['status'] == 'ok').sum()
    n_fail = (df_out['status'] == 'failed').sum()
    print(f"结果已保存: {args.output}")
    print(f"检索合格: {n_ok} / {len(df_out)} 靶点; 检索失败: {n_fail} 靶点")


if __name__ == '__main__':
    main()
