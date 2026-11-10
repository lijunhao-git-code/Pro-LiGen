"""Shared strict chemistry and atomic provenance utilities for this experiment."""
from datetime import datetime,timezone
from functools import lru_cache
import hashlib,json,os,re
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PROJECT=Path('/root/PG-TALIRAGEN/PG-TALIRAGEN')
PROPERTIES={'ring':dict(label='RDKit RingCount',delta=-1,tolerance=0),
            'logp':dict(label='Crippen LogP',delta=-1.,tolerance=.2),
            'tpsa':dict(label='TPSA',delta=-20.,tolerance=5.,units='Angstrom squared')}


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with temp.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)
def read_jsonl(path):return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]
def canonical(mol):
    from rdkit import Chem
    return Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)


def seed_component(smiles):
    from rdkit import Chem
    mol=Chem.MolFromSmiles(smiles)
    if mol is None:raise ValueError('Invalid seed')
    parts=Chem.GetMolFrags(mol,asMols=True)
    carbon=[p for p in parts if any(a.GetAtomicNum()==6 for a in p.GetAtoms())]
    other=[p for p in parts if not any(a.GetAtomicNum()==6 for a in p.GetAtoms())]
    if len(carbon)!=1 or any(p.GetNumAtoms()!=1 for p in other):raise ValueError('Ambiguous seed components')
    return canonical(carbon[0])


@lru_cache(maxsize=100000)
def properties(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors,QED
    sascorer=sa_scorer()
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumHeavyAtoms()==0:raise ValueError('Invalid or empty molecule')
    return dict(canonical=canonical(mol),logp=float(Descriptors.MolLogP(mol)),tpsa=float(Descriptors.TPSA(mol)),
        QED=float(QED.qed(mol)),SA_raw=float(sascorer.calculateScore(mol)),MW=float(Descriptors.MolWt(mol)),
        HeavyAtoms=mol.GetNumHeavyAtoms(),RC=int(Descriptors.RingCount(mol)),fragments=len(Chem.GetMolFrags(mol)),
        contains_carbon=any(a.GetAtomicNum()==6 for a in mol.GetAtoms()))


@lru_cache(maxsize=1)
def sa_scorer():
    from rdkit import RDConfig
    path=Path(RDConfig.RDContribDir)/'SA_Score'
    if not all((path/name).is_file() for name in ['sascorer.py','fpscores.pkl.gz']):
        raise RuntimeError('Standard RDKit SA scorer/data not available; no fallback')
    sys.path.insert(0,str(path))
    import sascorer
    sascorer.readFragmentScores(str(path/'fpscores.pkl.gz'))
    return sascorer


def in_band(value,center,tolerance):return abs(float(value)-float(center))<=float(tolerance)+1e-8


def evaluate_smiles(smiles,entry,turn=4,position=1):
    row=dict(turn=turn,candidate_position=position,smiles=smiles,valid=False,property_hit=False,eligible=False)
    if not smiles:return dict(row,reason='missing_tagged_proposal')
    try:
        row.update(properties(smiles),valid=True)
        row['property_value']=row['RC' if entry['property']=='ring' else entry['property']]
        row['target_error']=row['property_value']-entry['target_center']
        row['delta_property']=row['property_value']-entry['seed_property']
        row['property_hit']=in_band(row['property_value'],entry['target_center'],entry['tolerance'])
        row['copy_seed']=row['canonical']==entry['seed_canonical']
        row['copy_exemplar']=row['canonical'] in set(entry['exemplars'])
        if row['fragments']!=1:row['reason']='multiple_fragments'
        elif entry['property']!='ring' and not row['contains_carbon']:row['reason']='no_carbon'
        elif row['copy_seed'] or row['copy_exemplar']:row['reason']='exact_input_copy'
        elif not row['property_hit']:row['reason']='outside_property_band'
        else:row.update(eligible=True,reason='eligible')
    except (ValueError,RuntimeError):row['reason']='invalid_smiles'
    return row


def candidate_records(content,entry,turn):
    tags=re.findall(r'<smiles>\s*(.*?)\s*</smiles>',content or '',flags=re.I|re.S)
    rows=[evaluate_smiles(tags[i].strip() if i<len(tags) else None,entry,turn,i+1) for i in range(3)]
    return rows,max(0,len(tags)-3)


def choose(rows,policy='best_QED'):
    ordered=sorted(rows,key=lambda r:(r['turn'],r['candidate_position']))
    eligible=[r for r in ordered if r['eligible']]
    if not eligible:return None
    if policy=='first_eligible':return eligible[0]
    unique={r['canonical']:r for r in eligible}
    return min(unique.values(),key=lambda r:(-r['QED'],r['SA_raw'],r['canonical']))


def feedback(rows,entry):
    fields=['candidate_position','canonical','valid','property_value','delta_property','target_error','property_hit','eligible','reason','QED','SA_raw']
    actual=[{k:r.get(k) for k in fields} for r in rows]
    return ('\n【程序实际计算的上一轮结果】\n'+json.dumps(actual,ensure_ascii=False)+
        f'\n当前目标仅为{PROPERTIES[entry["property"]]["label"]}位于[{entry["target_lower"]:.8f}, {entry["target_upper"]:.8f}]。'
        '\n以上由RDKit与标准SA scorer计算。上一轮合格候选会保留；请再提出3个候选，不复制种子或所给范例。未执行对接，不要编造Vina或实验结果。')
