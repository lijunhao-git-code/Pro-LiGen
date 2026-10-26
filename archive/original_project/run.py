import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Data, Batch
from rdkit import Chem
import numpy as np
from typing import List, Union

# ------------------------------
# 1. 特征提取函数（必须与训练时完全相同）
# ------------------------------
def atom_features(atom):
    """提取原子特征，返回长度为 84 的向量"""
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
    if hybrid == Chem.rdchem.HybridizationType.SP:
        hybrid_onehot[0] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP2:
        hybrid_onehot[1] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3:
        hybrid_onehot[2] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D:
        hybrid_onehot[3] = 1.0
    elif hybrid == Chem.rdchem.HybridizationType.SP3D2:
        hybrid_onehot[4] = 1.0
    else:
        hybrid_onehot[5] = 1.0

    aromatic = [1.0] if atom.GetIsAromatic() else [0.0]

    h_count = atom.GetTotalNumHs()
    h_onehot = [0.0] * 5
    if h_count <= 4:
        h_onehot[h_count] = 1.0

    return atomic_onehot + degree_onehot + charge_onehot + chiral + hybrid_onehot + aromatic + h_onehot

def bond_features(bond):
    """提取键特征，返回长度为 9 的向量"""
    bt = bond.GetBondType()
    bond_onehot = [0.0] * 4
    if bt == Chem.rdchem.BondType.SINGLE:
        bond_onehot[0] = 1.0
    elif bt == Chem.rdchem.BondType.DOUBLE:
        bond_onehot[1] = 1.0
    elif bt == Chem.rdchem.BondType.TRIPLE:
        bond_onehot[2] = 1.0
    elif bt == Chem.rdchem.BondType.AROMATIC:
        bond_onehot[3] = 1.0

    conj = [1.0] if bond.GetIsConjugated() else [0.0]
    ring = [1.0] if bond.IsInRing() else [0.0]
    stereo = [0.0, 0.0, 0.0]
    stereo_type = bond.GetStereo()
    if stereo_type == Chem.rdchem.BondStereo.STEREOZ:
        stereo[0] = 1.0
    elif stereo_type == Chem.rdchem.BondStereo.STEREOE:
        stereo[1] = 1.0
    else:
        stereo[2] = 1.0

    return bond_onehot + conj + ring + stereo   # 4+1+1+3 = 9

def mol_to_graph(smiles: str):
    """将 SMILES 转换为 PyG Data 对象"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 节点特征
    atom_feats = [atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    # 边索引和边特征
    edges = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
        bf = bond_features(bond)
        edge_feats.append(bf)
        edge_feats.append(bf.copy())
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)   # [E, 9]
    else:
        # 无键分子（如单个原子）
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 9), dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

# ------------------------------
# 2. GIN 模型定义（与训练时相同）
# ------------------------------
class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, out_dim=256, num_layers=5):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                mlp = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            else:
                mlp = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            self.convs.append(GINConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.pool = global_mean_pool

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.pool(x, batch)
        return x

# ------------------------------
# 3. 推理类：加载模型并生成嵌入向量
# ------------------------------
class MolecularEmbedder:
    def __init__(self, model_path: str, device: Union[str, torch.device] = None):
        """
        Args:
            model_path: 训练好的模型权重文件路径（best_gin_model.pth）
            device: 运行设备，默认自动检测 cuda 或 cpu
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # 通过一个示例分子获取输入维度（特征维度固定为84）
        sample_graph = mol_to_graph("CC")
        if sample_graph is None:
            raise RuntimeError("无法生成样例图，请检查 RDKit 安装")
        in_dim = sample_graph.x.size(1)

        self.model = GINEncoder(in_dim=in_dim, hidden_dim=256, out_dim=256, num_layers=5)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def embed(self, smiles_list: List[str], return_numpy: bool = True) -> np.ndarray:
        """
        将 SMILES 列表转换为嵌入向量

        Args:
            smiles_list: SMILES 字符串列表
            return_numpy: 若 True 返回 numpy 数组，否则返回 torch.Tensor

        Returns:
            嵌入矩阵，形状 (有效分子数, 256)
        """
        # 转换所有 SMILES 为图，跳过无效的
        graphs = []
        valid_indices = []
        for i, smi in enumerate(smiles_list):
            g = mol_to_graph(smi)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i)

        if len(graphs) == 0:
            print("Warning: No valid SMILES provided.")
            return np.empty((0, 256)) if return_numpy else torch.empty((0, 256))

        # 批量合并
        batch = Batch.from_data_list(graphs)
        batch = batch.to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)  # [N, 256]

        if return_numpy:
            embeddings = embeddings.cpu().numpy()
        return embeddings

    def embed_single(self, smiles: str, return_numpy: bool = True) -> Union[np.ndarray, torch.Tensor]:
        """
        生成单个分子的嵌入向量

        Args:
            smiles: SMILES 字符串
            return_numpy: 若 True 返回 numpy 数组，否则返回 torch.Tensor

        Returns:
            形状 (256,) 的向量，若分子无效则返回 None
        """
        graph = mol_to_graph(smiles)
        if graph is None:
            print(f"Invalid SMILES: {smiles}")
            return None
        batch = Batch.from_data_list([graph])
        batch = batch.to(self.device)
        with torch.no_grad():
            emb = self.model(batch).squeeze(0)  # (256,)
        if return_numpy:
            emb = emb.cpu().numpy()
        return emb

# ------------------------------
# 使用示例
# ------------------------------
if __name__ == "__main__":
    # 初始化嵌入器（请修改为实际模型路径）
    embedder = MolecularEmbedder("best_gin_model.pth")

    # 示例 SMILES
    smiles_list = [
        "CC(=O)Oc1ccccc1C(=O)O",   # 阿司匹林
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # 咖啡因
        "c1ccccc1"                 # 苯
    ]

    # 批量生成嵌入
    embeddings = embedder.embed(smiles_list, return_numpy=True)
    print(f"嵌入矩阵形状: {embeddings.shape}")  # (3, 256)

    # 单个分子生成
    emb_single = embedder.embed_single("CCCC")
    if emb_single is not None:
        print(f"正丁烷嵌入向量前5维: {emb_single[:5]}")