"""Offline molecular descriptors and structural features; never sends API requests."""
from functools import lru_cache
import hashlib,json,math
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import Descriptors,QED,rdFingerprintGenerator
from common import ROOT,sa_scorer
RDLogger.DisableLog('rdApp.*')
FP=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=False)


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [clean(v) for v in value]
    if isinstance(value,(np.integer,np.bool_)):return value.item()
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    return value


def save_json(path,value):
    from common import dump
    dump(path,clean(value))


def csv(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(path,index=False,encoding='utf-8-sig',float_format='%.15g')


@lru_cache(maxsize=100000)
def describe(smiles):
    mol=Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None or mol.GetNumHeavyAtoms()==0:return None
    return dict(canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True),
        logp=float(Descriptors.MolLogP(mol)),tpsa=float(Descriptors.TPSA(mol)),QED=float(QED.qed(mol)),
        SA_raw=float(sa_scorer().calculateScore(mol)),MW=float(Descriptors.MolWt(mol)),
        HeavyAtoms=mol.GetNumHeavyAtoms(),RC=int(Descriptors.RingCount(mol)),
        fragments=len(Chem.GetMolFrags(mol)),contains_carbon=any(a.GetAtomicNum()==6 for a in mol.GetAtoms()))


@lru_cache(maxsize=100000)
def fingerprint(smiles):return FP.GetFingerprint(Chem.MolFromSmiles(smiles))


def structure(candidate,seed):
    a,b=describe(candidate),describe(seed)
    return dict(tanimoto=float(DataStructs.TanimotoSimilarity(fingerprint(candidate),fingerprint(seed))),
        delta_MW=a['MW']-b['MW'],delta_HeavyAtoms=a['HeavyAtoms']-b['HeavyAtoms'],delta_RC=a['RC']-b['RC'])


def mcs_key(seed,candidate):return hashlib.sha256((seed+'\0'+candidate).encode()).hexdigest()


def choose_independent(rows,policy='best_QED'):
    eligible=[r for r in sorted(rows,key=lambda r:(r['turn'],r['candidate_position'])) if r['eligible']]
    if not eligible:return None
    if policy=='first_eligible':return eligible[0]
    return min({r['canonical']:r for r in eligible}.values(),key=lambda r:(-r['QED'],r['SA_raw'],r['canonical']))


def policy_support(rows,policy):
    eligible=[r for r in rows if r['eligible']]
    if not eligible:return []
    if policy=='random_unique_expected':return list({r['canonical']:r for r in eligible}.values())
    return [choose_independent(rows,policy)]
