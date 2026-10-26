"""GIN graph features and frozen encoder restored from retained experiment sources.

Feature functions are extracted unchanged from the archived trainer. The graph
builder and GINEncoder are the exact source functions saved with Study B.
See provenance/gin_source_restoration.json. Importing this module does not train
a model, call an API or perform docking.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.nn import GINConv, global_mean_pool


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
