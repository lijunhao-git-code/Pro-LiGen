"""Shared paths, strict chemistry scoring, JSON and provenance utilities."""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

PROJECT = Path('/root/PG-TALIRAGEN/PG-TALIRAGEN')
EXP = PROJECT / 'mw_experiments'
OUT = Path(__file__).resolve().parent

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def now():
    return datetime.now(timezone.utc).isoformat()

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def canon(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty molecule')
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

@lru_cache(maxsize=1)
def scorer():
    from rdkit import RDConfig
    location = Path(RDConfig.RDContribDir) / 'SA_Score'
    if not all((location / name).is_file() for name in ('sascorer.py', 'fpscores.pkl.gz')):
        raise RuntimeError('Standard SA scorer or fragment database unavailable; no fallback allowed')
    sys.path.insert(0, str(location))
    import sascorer
    sascorer.readFragmentScores(str(location / 'fpscores.pkl.gz'))
    return sascorer, {name: {'path': str(location / name), 'sha256': sha(location / name)}
                      for name in ('sascorer.py', 'fpscores.pkl.gz')}

def properties(smiles):
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED
    sa, _ = scorer()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError('Invalid or empty molecule')
    value = float(sa.calculateScore(mol))
    return {'canonical': canon(smiles), 'RC': int(Descriptors.RingCount(mol)),
            'QED': float(QED.qed(mol)), 'SA_raw': value, 'SA_norm': (10. - value) / 9.,
            'MW': float(Descriptors.MolWt(mol)), 'HeavyAtoms': int(mol.GetNumHeavyAtoms()),
            'fragments': len(Chem.GetMolFrags(mol))}

def extract_smiles(text):
    """First parseable SMILES in TEXT ORDER, never ranked by computed properties."""
    candidates = [(m.start(), m.group(1).strip()) for m in re.finditer(
        r'<smiles>\s*(.*?)\s*</smiles>', text or '', flags=re.I | re.S)]
    candidates += [(m.start(), m.group(1).strip()) for m in re.finditer(
        r'(?m)^\s*([A-Za-z0-9@\[\]\(\)/\\#%=.+:\-]{10,})\s*$', text or '')]
    for _, smiles in sorted(candidates, key=lambda x: x[0]):
        try:
            canon(smiles)
            return smiles
        except (ValueError, RuntimeError):
            continue
    return None

def candidate_records(text, turn, entry):
    """Three fixed tagged proposal positions per design turn, including invalids."""
    tags = re.findall(r'<smiles>\s*(.*?)\s*</smiles>', text or '', flags=re.I | re.S)
    forbidden = {canon(entry['seed_smiles'])} | {canon(s) for s in entry['exemplars']}
    records = []
    for index in range(3):
        smi = tags[index].strip() if index < len(tags) else None
        record = dict(turn=turn, candidate_position=index+1, smiles=smi, valid=False, eligible=False)
        if not smi:
            record['reason'] = 'missing_tagged_proposal'
        else:
            try:
                record.update(properties(smi), valid=True)
                record['copy'] = record['canonical'] in forbidden
                if record['fragments'] != 1:
                    record['reason'] = 'multiple_fragments'
                elif record['copy']:
                    record['reason'] = 'exact_input_copy'
                elif record['RC'] != entry['target_rc']:
                    record['reason'] = 'wrong_ring_count'
                else:
                    record.update(eligible=True, reason='eligible')
            except (ValueError, RuntimeError):
                record['reason'] = 'invalid_smiles'
        records.append(record)
    return records, max(0, len(tags)-3)

def select_candidate(records):
    """Prelocked selector; never consult docking or model-reported scores."""
    eligible = {r['canonical']: r for r in records if r['eligible']}
    if not eligible:
        return None
    return min(eligible.values(), key=lambda r: (-r['QED'], r['SA_raw'], r['canonical']))

def optimization_feedback(records, entry):
    proposals = [dict(position=r['candidate_position'], canonical=r.get('canonical'),
                      valid=r['valid'], eligible=r['eligible'], reason=r['reason'],
                      RC=r.get('RC'), QED=r.get('QED'), SA_raw=r.get('SA_raw')) for r in records]
    return ('\n【程序实际计算的上一轮结果】\n'+json.dumps(proposals, ensure_ascii=False)+
            '\n以上由 RDKit 和标准 SA scorer 实际计算。未计算对接。请保持环数恰好为 '+
            str(entry['target_rc'])+'；所有有效且满足条件的上一轮候选会保留到最终程序排序。')

def verify_locked():
    lock = json.loads((OUT / 'locked.json').read_text())
    for relative, expected in lock['files'].items():
        if sha(OUT / relative) != expected:
            raise RuntimeError('Locked experiment file changed: ' + relative)
    for filename, expected in lock['readonly_inputs'].items():
        if sha(filename) != expected:
            raise RuntimeError('Read-only experiment input changed: ' + filename)
    return json.loads((OUT / 'protocol.json').read_text())
