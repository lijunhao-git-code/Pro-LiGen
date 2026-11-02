"""
train_gin.py — 训练 GIN 模型（InfoNCE 三元组损失）

用法：
    python mw_experiments/train_gin.py --csv mw_experiments/triplets/triplets_mw.csv --output mw_experiments/models/best_gin_mw.pth

复用 finetune.py 的模型架构和训练逻辑，加上命令行参数和训练日志记录。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import warnings
import csv as csv_module
from pathlib import Path
warnings.filterwarnings('ignore')


# ============================
# 特征提取（与 finetune.py 完全一致）
# ============================
def atom_features(atom):
    atomic_num = atom.GetAtomicNum()
    atomic_onehot = [0.0] * 60
    if 1 <= atomic_num <= 60:
        atomic_onehot[atomic_num - 1] = 1.0

    degree = atom.GetDegree()
    degree_onehot = [0.0] * 6
    if degree <= 5:
        degree_onehot[degree] = 1.0

    formal_charge = atom.GetFormalCharge()
    charge_onehot = [0.0] * 5
    if -2 <= formal_charge <= 2:
        charge_onehot[formal_charge + 2] = 1.0

    chiral = [1.0] if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED else [0.0]

    hybrid = atom.GetHybridization()
    hybrid_onehot = [0.0] * 6
    if hybrid == Chem.rdchem.HybridizationType.SP:      hybrid_onehot[0] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP2:    hybrid_onehot[1] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3:    hybrid_onehot[2] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D:   hybrid_onehot[3] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D2:  hybrid_onehot[4] = 1.0
    else:                                                 hybrid_onehot[5] = 1.0

    aromatic = [1.0] if atom.GetIsAromatic() else [0.0]

    h_count = atom.GetTotalNumHs()
    h_onehot = [0.0] * 5
    if h_count <= 4:
        h_onehot[h_count] = 1.0

    return atomic_onehot + degree_onehot + charge_onehot + chiral + hybrid_onehot + aromatic + h_onehot


def bond_features(bond):
    bt = bond.GetBondType()
    bond_onehot = [0.0] * 4
    if bt == Chem.rdchem.BondType.SINGLE:    bond_onehot[0] = 1.0
    elif bt == Chem.rdchem.BondType.DOUBLE:  bond_onehot[1] = 1.0
    elif bt == Chem.rdchem.BondType.TRIPLE:  bond_onehot[2] = 1.0
    elif bt == Chem.rdchem.BondType.AROMATIC: bond_onehot[3] = 1.0

    conj = [1.0] if bond.GetIsConjugated() else [0.0]
    ring = [1.0] if bond.IsInRing() else [0.0]
    stereo = [0.0, 0.0, 0.0]
    if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:   stereo[0] = 1.0
    elif bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE: stereo[1] = 1.0
    else:                                                     stereo[2] = 1.0

    return bond_onehot + conj + ring + stereo


def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j]); edges.append([j, i])
        bf = bond_features(bond)
        edge_feats.append(bf); edge_feats.append(bf.copy())

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 9), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ============================
# GIN 模型
# ============================
class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            else:
                mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
        return self.pool(x, batch)


# ============================
# 数据集
# ============================
class TripletDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        for col in ['anchor_smiles', 'positive_smiles', 'negative_smiles']:
            if col not in self.df.columns:
                raise ValueError(f"CSV 缺少列: {col}")
        self.anchors = self.df['anchor_smiles'].tolist()
        self.positives = self.df['positive_smiles'].tolist()
        self.negatives = self.df['negative_smiles'].tolist()

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        a = mol_to_graph(self.anchors[idx])
        p = mol_to_graph(self.positives[idx])
        n = mol_to_graph(self.negatives[idx])
        if a is None or p is None or n is None:
            return None
        return (a, p, n)


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    anchors, positives, negatives = zip(*batch)
    return list(anchors), list(positives), list(negatives)


def pyg_collate(anchors, positives, negatives):
    return Batch.from_data_list(anchors), Batch.from_data_list(positives), Batch.from_data_list(negatives)


# ============================
# InfoNCE Loss
# ============================
def infonce_loss(anchor_emb, positive_emb, negative_emb, temperature=0.1):
    B = anchor_emb.size(0)
    all_candidates = torch.cat([positive_emb, negative_emb], dim=0)
    sim = torch.matmul(anchor_emb, all_candidates.T) / temperature
    labels = torch.arange(B, device=anchor_emb.device)
    return F.cross_entropy(sim, labels)


# ============================
# 训练 / 验证
# ============================
def train_one_epoch(model, dataloader, optimizer, device, temperature):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if batch is None: continue
        anchors, positives, negatives = batch
        if len(anchors) == 0: continue
        a_batch, p_batch, n_batch = pyg_collate(anchors, positives, negatives)
        a_batch, p_batch, n_batch = a_batch.to(device), p_batch.to(device), n_batch.to(device)

        loss = infonce_loss(model(a_batch), model(p_batch), model(n_batch), temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1
    return total_loss / max(num_batches, 1)


def validate(model, dataloader, device, temperature):
    model.eval()
    total_loss, num_batches = 0.0, 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            if batch is None: continue
            anchors, positives, negatives = batch
            if len(anchors) == 0: continue
            a_batch, p_batch, n_batch = pyg_collate(anchors, positives, negatives)
            a_batch, p_batch, n_batch = a_batch.to(device), p_batch.to(device), n_batch.to(device)
            loss = infonce_loss(model(a_batch), model(p_batch), model(n_batch), temperature)
            total_loss += loss.item()
            num_batches += 1
    return total_loss / max(num_batches, 1)


# ============================
# 主程序
# ============================
def main():
    parser = argparse.ArgumentParser(description='Train GIN encoder with InfoNCE triplet loss')
    parser.add_argument('--csv', type=str, required=True, help='三元组 CSV 路径')
    parser.add_argument('--output', type=str, required=True, help='模型保存路径 (.pth)')
    parser.add_argument('--log', type=str, default=None, help='训练日志 CSV 路径')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--out_dim', type=int, default=256)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=0)
    args = parser.parse_args()

    # 默认日志路径
    if args.log is None:
        log_path = str(Path(args.output).with_suffix('.csv'))
    else:
        log_path = args.log

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Triplet CSV: {args.csv}")
    print(f"Model output: {args.output}")
    print(f"Log: {log_path}")

    # 数据集
    dataset = TripletDataset(args.csv)
    total = len(dataset)
    train_len = int(0.8 * total)
    val_len = total - train_len
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_len, val_len])
    print(f"Total: {total} | Train: {train_len} | Val: {val_len}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # 模型
    sample = mol_to_graph("CC")
    in_dim = sample.x.size(1)
    print(f"Input dim: {in_dim}")
    model = GINEncoder(in_dim, args.hidden_dim, args.out_dim, args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 训练循环 — 每个 epoch 增量写入日志
    best_val_loss = float('inf')
    patience_counter = 0

    # 打开日志文件，边训练边写
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, 'w', newline='')
    log_writer = csv_module.DictWriter(log_file, fieldnames=['epoch', 'train_loss', 'val_loss'])
    log_writer.writeheader()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.temperature)
        val_loss = validate(model, val_loader, device, args.temperature)
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # 立即写入 CSV
        log_writer.writerow({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        log_file.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.output)
            print(f"  -> Best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    log_file.close()
    print(f"Training log saved to {log_path}")
    print("Done.")


if __name__ == '__main__':
    main()
