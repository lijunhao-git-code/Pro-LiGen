"""
build_triplets.py — 构建属性导向的三元组数据集（支持 MW / xlogp / TPSA / Ring_Count）

用法：
    python mw_experiments/build_triplets.py --property MW --k 10
    python mw_experiments/build_triplets.py --property xlogp --k 10
    python mw_experiments/build_triplets.py --property TPSA --k 10
    python mw_experiments/build_triplets.py --property Ring_Count --k 10

输入：csv_folder/train.csv
输出：mw_experiments/triplets/triplets_{property}.csv
"""

import pandas as pd
import numpy as np
import argparse
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors
from sklearn.metrics.pairwise import cosine_similarity


# ---------- 属性计算 ----------
def compute_property(smiles: str, prop_name: str) -> float | None:
    """从 SMILES 计算指定属性值，失败返回 None"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    prop_name = prop_name.lower()
    if prop_name == 'mw':
        return Descriptors.MolWt(mol)
    elif prop_name == 'xlogp':
        # 由 train.csv 的 xlogp 列提供，这里给出 fallback
        return Descriptors.MolLogP(mol)
    elif prop_name == 'tpsa':
        return Descriptors.TPSA(mol)
    elif prop_name == 'ring_count':
        return Descriptors.RingCount(mol)
    else:
        raise ValueError(f"不支持的属性: {prop_name}")


# ---------- Morgan 指纹 ----------
def smiles_to_embedding(smiles: str, radius: int = 2, nBits: int = 2048) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nBits)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(fp, dtype=np.float32)


# ---------- 三元组构建 ----------
def build_triplet_dataset(
    csv_path: str,
    prop_name: str,
    k: int = 10,
    output_path: str = None,
    direction: str = 'auto',
):
    """
    构建 (anchor, positive, negative) 三元组。

    Parameters
    ----------
    csv_path : 训练集 CSV 路径（含 SMILES 列 + 可选 xlogp/polararea）
    prop_name : 属性名 (MW / xlogp / TPSA / Ring_Count)
    k : 结构近邻数（论文 K=10）
    output_path : 输出路径，默认自动生成
    direction : 'lower' / 'higher' / 'auto'
                auto: MW → lower, xlogp → lower, TPSA → 视情况, Ring_Count → lower
    """
    if output_path is None:
        from pathlib import Path
        output_dir = Path(__file__).resolve().parent / 'triplets'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f'triplets_{prop_name.lower()}.csv')

    # 确定排序方向
    if direction == 'auto':
        if prop_name.lower() in ('mw', 'xlogp', 'ring_count'):
            direction = 'lower'
        else:
            direction = 'lower'  # 默认越低越好

    df = pd.read_csv(csv_path)
    smiles_list = df['smiles'].tolist()  # train.csv 列名是小写 smiles

    # ----- 属性值获取 -----
    prop_values = []
    # 尝试从 CSV 列直接读取
    col_map = {'xlogp': 'xlogp', 'tpsa': 'polararea', 'mw': None, 'ring_count': None}
    csv_col = col_map.get(prop_name.lower())

    for i, smi in enumerate(smiles_list):
        if csv_col and csv_col in df.columns:
            val = df[csv_col].iloc[i]
            if pd.isna(val):
                val = compute_property(smi, prop_name)
        else:
            val = compute_property(smi, prop_name)
        prop_values.append(val)

    # ----- Morgan 指纹 -----
    embeddings = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        emb = smiles_to_embedding(smi)
        if emb is not None and prop_values[i] is not None:
            embeddings.append(emb)
            valid_indices.append(i)

    if len(embeddings) < k + 1:
        raise ValueError(f"有效分子数量 {len(embeddings)} 不足 {k+1}")

    embeddings = np.array(embeddings)
    valid_smiles = [smiles_list[i] for i in valid_indices]
    valid_props = [prop_values[i] for i in valid_indices]
    n = len(valid_smiles)

    # L2 归一化 + 余弦相似度
    norm_emb = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    sim_mat = np.dot(norm_emb, norm_emb.T)

    # ----- 构建三元组 -----
    triplets = []
    for i in range(n):
        sim = sim_mat[i]
        neighbor_idx = np.argsort(sim)[::-1]          # 相似度从高到低
        neighbor_idx = [idx for idx in neighbor_idx if idx != i][:k]
        if len(neighbor_idx) < k:
            continue

        neighbor_props = [valid_props[idx] for idx in neighbor_idx]
        if direction == 'lower':
            pos_idx = neighbor_idx[np.argmin(neighbor_props)]
            neg_idx = neighbor_idx[np.argmax(neighbor_props)]
        else:
            pos_idx = neighbor_idx[np.argmax(neighbor_props)]
            neg_idx = neighbor_idx[np.argmin(neighbor_props)]

        if pos_idx == neg_idx:
            continue

        triplets.append({
            'anchor_smiles': valid_smiles[i],
            'positive_smiles': valid_smiles[pos_idx],
            'negative_smiles': valid_smiles[neg_idx],
            f'anchor_{prop_name.lower()}': valid_props[i],
            f'positive_{prop_name.lower()}': valid_props[pos_idx],
            f'negative_{prop_name.lower()}': valid_props[neg_idx],
        })

    triplet_df = pd.DataFrame(triplets)
    triplet_df.to_csv(output_path, index=False)
    print(f"属性: {prop_name} | 方向: {direction} | K: {k}")
    print(f"有效分子: {n} | 三元组: {len(triplets)}")
    print(f"输出: {output_path}")
    return triplet_df


# ---------- CLI ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='构建属性导向三元组数据集')
    parser.add_argument('--property', type=str, default='MW',
                        choices=['MW', 'xlogp', 'TPSA', 'Ring_Count'],
                        help='目标属性名称')
    parser.add_argument('--k', type=int, default=10,
                        help='结构近邻数 K（论文设 10）')
    parser.add_argument('--direction', type=str, default='auto',
                        choices=['auto', 'lower', 'higher'],
                        help='属性优化方向')
    parser.add_argument('--csv', type=str, default='csv_folder/train.csv',
                        help='训练集 CSV 路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出路径（默认自动）')
    args = parser.parse_args()

    build_triplet_dataset(
        csv_path=args.csv,
        prop_name=args.property,
        k=args.k,
        output_path=args.output,
        direction=args.direction,
    )
