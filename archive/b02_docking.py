"""Independent B02 chemistry/calibration; no API credentials or model calls.

Run only inside the new B02 directory. Original experiments are read-only.
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = Path('/root/PG-TALIRAGEN/PG-TALIRAGEN')
EXP = PROJECT / 'mw_experiments'
B = EXP / 'gin_downstream_20261011_375'
assert ROOT.name.startswith('neurocomputing-b02-') and ROOT.parent == Path('/tmp')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def dump(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(url, p):
    import requests
    if not p.exists():
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        assert len(r.content) > 50
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
    return p


def active(smiles):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError('invalid parent molecule')
    frags = Chem.GetMolFrags(mol, asMols=True)
    organic = [m for m in frags if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    if len(organic) != 1:
        raise ValueError('ambiguous number of organic fragments')
    selected = organic[0]
    if any(m is not selected and m.GetNumAtoms() != 1 for m in frags):
        raise ValueError('non-atomic inorganic component requires review')
    return Chem.MolToSmiles(selected, isomericSmiles=True)


def reference_mol(comp, ligand_pdb, out):
    """Explicit CCD graph + complete named experimental heavy-atom coordinates."""
    import gemmi
    from rdkit import Chem
    from rdkit.Geometry import Point3D
    p = fetch('https://files.rcsb.org/ligands/download/'+comp+'.cif', ROOT/'sources'/('CCD_'+comp+'.cif'))
    block = gemmi.cif.read(str(p)).sole_block()
    atoms = block.find('_chem_comp_atom.', ['atom_id', 'type_symbol', 'charge'])
    bonds = block.find('_chem_comp_bond.', ['atom_id_1', 'atom_id_2', 'value_order'])
    coords = {}
    for line in Path(ligand_pdb).read_text().splitlines():
        if line.startswith(('ATOM  ', 'HETATM')) and line[16] in (' ', 'A'):
            name = line[12:16].strip()
            if name in coords:
                raise ValueError('duplicate atom name/alternate conformation')
            coords[name] = tuple(float(line[s:e]) for s, e in [(30, 38), (38, 46), (46, 54)])
    rw, ids, names = Chem.RWMol(), {}, []
    for row in atoms:
        name, element, charge = map(str, row)
        name = gemmi.cif.as_string(name)
        if element in ('H', 'D'):
            continue
        if name not in coords:
            raise ValueError('missing experimental heavy atom '+name)
        atom = Chem.Atom(element.title())
        atom.SetFormalCharge(int(charge) if charge not in ('?', '.') else 0)
        atom.SetProp('CCD_atom_id', name)
        ids[name] = rw.AddAtom(atom)
        names.append(name)
    types = {'SING': Chem.BondType.SINGLE, 'DOUB': Chem.BondType.DOUBLE,
             'TRIP': Chem.BondType.TRIPLE, 'AROM': Chem.BondType.AROMATIC}
    for row in bonds:
        a, b, order = map(str, row)
        a, b = gemmi.cif.as_string(a), gemmi.cif.as_string(b)
        if a in ids and b in ids:
            rw.AddBond(ids[a], ids[b], types[order])
    mol = rw.GetMol()
    Chem.SanitizeMol(mol)
    conf = Chem.Conformer(len(names))
    for i, name in enumerate(names):
        conf.SetAtomPosition(i, Point3D(*coords[name]))
    mol.AddConformer(conf)
    Chem.AssignStereochemistryFrom3D(mol)
    assert mol.GetNumHeavyAtoms() == len(names)
    with Chem.SDWriter(str(out)) as w:
        w.write(mol)
    return mol


def prepare():
    assert not (ROOT/'receptors.json').exists(), 'Already prepared; preserve manifest'
    decisions = json.loads((B/'receptor_audit/decisions.json').read_text())
    targets = list(csv.DictReader((B/'inputs/targets.csv').open()))
    refs, audit = {}, []
    protected = [PROJECT.parent.parent/'Pro_LiGAN/achemso-demo.tex']
    protected = [Path('/root/Pro_LiGAN/achemso-demo.tex'), B/'locked.json', B/'protocol.json',
                 B/'inputs/targets.csv', B/'inputs/prompts.jsonl', B/'receptor_audit/decisions.json']
    protected += list((EXP/'generated').glob('exp30rc25_*.txt'))
    protected += list((B/'results').glob('*.json'))
    protected += list((B/'receptors').glob('*/*'))
    dump(ROOT/'protected_before.json', {str(p): sha(p) for p in protected if p.is_file()})
    config = dict(created_utc=now(), seed=42, exhaustiveness=16, n_poses=10, cpu=1, workers=8,
                  timeout_seconds=600, sf_name='vina', top1_rmsd_cutoff_A=2.0,
                  metal_targets_excluded_from_primary=True,
                  parent_not_overwritten=True, salt_rule='unique carbon-containing fragment; only atomic counterions',
                  ligand_prep='input formal charge; RDKit AddHs, Embed42/fallback random42, MMFF500, Meeko',
                  ligand_protonation_enumeration=False, receptor_prep='reuse preaudited receptor; new: OpenBabel pH7.4; NAD explicitly retained',
                  no_rescue_parameter_tuning=True, rmsd='RDKit CalcRMS heavy symmetry-aware in place; no alignment')
    dump(ROOT/'docking_protocol.json', config)
    # Replacements selected by public identity/pocket evidence, not generated scores.
    additions = {'CHEMBL6055': ('8EDH', 'WGK'), 'CHEMBL2179': ('2XWD', 'LGS'),
                 'CHEMBL3712868': ('5ICM', 'F45')}
    for row in targets:
        t, u = row['chemblid'], row['uniprotid']
        old = decisions[t]
        recdir = ROOT/'receptors'/t
        recdir.mkdir(parents=True, exist_ok=True)
        entry = dict(chemblid=t, uniprotid=u, old_include=old['include_docking'])
        if not old['include_docking'] and t not in additions:
            entry.update(status='excluded_no_validated_matching_pocket', reason=old['reason'])
            audit.append(entry)
            continue
        try:
            if old['include_docking']:
                r = dict(old['receptor'])
                source = EXP/r['reference_pdb']
                qt = EXP/r['receptor_pdbqt']
                assert sha(source) == r['source_pdb_sha256'] and sha(qt) == r['pdbqt_sha256']
                for name in (source.name, 'receptor_clean.pdb', 'reference_ligand.pdb', 'receptor.pdbqt'):
                    shutil.copy2(qt.parent/name, recdir/name)
                r['receptor'] = str(recdir/'receptor.pdbqt')
                r['source_pdb'] = str(recdir/source.name)
            else:
                import numpy as np
                pdb, lig = additions[t]
                epath = fetch('https://data.rcsb.org/rest/v1/core/entry/'+pdb, ROOT/'sources'/(pdb+'_entry.json'))
                e = json.loads(epath.read_text())
                entities = []
                for eid in e['rcsb_entry_container_identifiers']['polymer_entity_ids']:
                    q = fetch('https://data.rcsb.org/rest/v1/core/polymer_entity/'+pdb+'/'+eid,
                              ROOT/'sources'/(pdb+'_entity_'+eid+'.json'))
                    entities.append(json.loads(q.read_text()))
                matching = [q for q in entities if any(a.get('database_accession') == u for a in
                            q['rcsb_polymer_entity_container_identifiers'].get('reference_sequence_identifiers', []))]
                assert matching and max(q['entity_poly']['rcsb_sample_sequence_length'] for q in matching) > 100
                chains = matching[0]['rcsb_polymer_entity_container_identifiers']['auth_asym_ids']
                source = fetch('https://files.rcsb.org/download/'+pdb+'.pdb', recdir/(pdb+'.pdb'))
                lines = source.read_text().splitlines()
                instances = {}
                for line in lines:
                    if line.startswith('HETATM') and line[17:20].strip() == lig and line[16] in (' ', 'A'):
                        instances.setdefault(line[21:27], []).append(line)
                pairs = []
                for c in chains:
                    protein = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in lines
                                        if l.startswith('ATOM  ') and l[21] == c and l[16] in (' ', 'A')])
                    for key, group in instances.items():
                        xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in group])
                        contacts = (np.sqrt(((xyz[:, None, :]-protein[None, :, :])**2).sum(axis=2)).min(axis=1) <= 4.5).sum()
                        pairs.append((-int(contacts), c, key))
                neg, chain, key = min(pairs)
                assert -neg >= 3
                anchor = instances[key]
                xyz = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in anchor])
                center = (xyz.min(axis=0)+xyz.max(axis=0))/2
                size = np.maximum(xyz.max(axis=0)-xyz.min(axis=0)+16, 20)
                atoms = [l for l in lines if l.startswith('ATOM  ') and l[21] == chain and l[16] in (' ', 'A')]
                (recdir/'receptor_clean.pdb').write_text('\n'.join(atoms)+'\nEND\n')
                (recdir/'reference_ligand.pdb').write_text('\n'.join(anchor)+'\nEND\n')
                run = subprocess.run(['/root/miniconda3/envs/ljh/bin/obabel', '-ipdb', str(recdir/'receptor_clean.pdb'),
                                      '-opdbqt', '-O', str(recdir/'receptor.pdbqt'), '-xr', '-p', '7.4'],
                                     capture_output=True, text=True, timeout=600)
                (recdir/'preparation.log').write_text(run.stdout+run.stderr)
                assert run.returncode == 0
                atomlines = [l for l in (recdir/'receptor.pdbqt').read_text().splitlines() if l.startswith(('ATOM  ', 'HETATM'))]
                if t == 'CHEMBL3712868':
                    nadlines = [l for l in lines if l.startswith('HETATM') and l[17:20].strip() == 'NAD'
                                and l[21] == chain and l[16] in (' ', 'A')]
                    assert nadlines
                    (recdir/'NAD.pdb').write_text('\n'.join(nadlines)+'\nEND\n')
                    from rdkit import Chem
                    from meeko import MoleculePreparation, PDBQTWriterLegacy
                    nad = reference_mol('NAD', recdir/'NAD.pdb', recdir/'NAD_reference.sdf')
                    setup = MoleculePreparation().prepare(Chem.AddHs(nad, addCoords=True))[0]
                    text, ok, error = PDBQTWriterLegacy().write_string(setup)
                    assert ok, error
                    atomlines += [l[:17]+'NAD'+l[20:] for l in text.splitlines() if l.startswith(('ATOM  ', 'HETATM'))]
                (recdir/'receptor.pdbqt').write_text('\n'.join(atomlines)+'\n')
                r = dict(chemblid=t, uniprotid=u, pdb_id=pdb, chains=[chain], cocrystal_ligand=lig,
                         anchor_instance=key, box_center=center.tolist(), box_size=size.tolist(),
                         receptor=str(recdir/'receptor.pdbqt'), source_pdb=str(source),
                         source_url='https://www.rcsb.org/structure/'+pdb, pdbqt_sha256=sha(recdir/'receptor.pdbqt'),
                         source_pdb_sha256=sha(source), matching_entity_lengths=[q['entity_poly']['rcsb_sample_sequence_length'] for q in matching],
                         anchor_contact_atoms=-neg, retained_cofactor='NAD' if t=='CHEMBL3712868' else None,
                         caveat='Public pocket verified; seed experimental binding pose not established; input protonation only')
            from vina import Vina
            v = Vina(cpu=1, seed=42, verbosity=0)
            v.set_receptor(r['receptor'])
            p = fetch('https://data.rcsb.org/rest/v1/core/entry/'+r['pdb_id'], ROOT/'sources'/(r['pdb_id']+'_entry.json'))
            e = json.loads(p.read_text())
            r['title'] = e['struct']['title']
            r['resolution_A'] = (e.get('rcsb_entry_info', {}).get('resolution_combined') or [None])[0]
            r['metal_atom_count'] = sum(l.startswith(('ATOM  ', 'HETATM')) and l[17:20].strip() in
                                      {'ZN','MG','MN','FE','CA','CU','CO','NI'} for l in Path(r['receptor']).read_text().splitlines())
            r['primary_nonmetal'] = r['metal_atom_count'] == 0
            r['pocket_seed_caveat'] = 'Pocket confirmed by a deposited ligand, not seed binding-mode validation'
            m = reference_mol(r['cocrystal_ligand'], recdir/'reference_ligand.pdb', recdir/'reference_ligand.sdf')
            from rdkit import Chem
            r['reference_canonical'] = Chem.MolToSmiles(m, isomericSmiles=True)
            r['reference_heavy_atoms'] = m.GetNumHeavyAtoms()
            refs[t] = r
            entry.update(status='ready_for_calibration', **r)
        except Exception as exc:
            entry.update(status='preparation_or_reference_failure', error_class=type(exc).__name__, reason=str(exc)[:400])
        audit.append(entry)
        dump(ROOT/'receptor_audit.json', audit)
        print('PREP',t,entry['status'],flush=True)
    dump(ROOT/'receptors.json', refs)
    dump(ROOT/'receptor_audit.json', audit)
    print('PREPARED',len(refs),'/25',flush=True)


def worker(task, calibration):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, rdMolAlign
    from meeko import MoleculePreparation, PDBQTWriterLegacy, PDBQTMolecule, RDKitMolCreate
    from vina import Vina
    RDLogger.DisableLog('rdApp.*')
    d = ROOT/('calibration' if calibration else 'docking')/task['job_id']
    p = json.loads((ROOT/'docking_protocol.json').read_text())
    d.mkdir(parents=True, exist_ok=True)
    out = dict(**task, started_utc=now(), status='technical_failure', vina=None)
    started = time.monotonic()
    try:
        mol = Chem.AddHs(Chem.MolFromSmiles(task['canonical']))
        if AllChem.EmbedMolecule(mol, randomSeed=42) != 0:
            if AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True) != 0:
                raise ValueError('3D embed failed twice')
        out['mmff_status'] = int(AllChem.MMFFOptimizeMolecule(mol, maxIters=500))
        setup = MoleculePreparation().prepare(mol)[0]
        qt, ok, err = PDBQTWriterLegacy().write_string(setup)
        assert ok, err
        (d/'ligand.pdbqt').write_text(qt)
        v = Vina(sf_name='vina',cpu=1,seed=42,verbosity=0)
        v.set_receptor(task['receptor'])
        v.set_ligand_from_file(str(d/'ligand.pdbqt'))
        v.compute_vina_maps(center=task['box_center'],box_size=task['box_size'])
        v.dock(exhaustiveness=16,n_poses=10)
        energies = v.energies(n_poses=10)
        assert len(energies)
        v.write_poses(str(d/'poses.pdbqt'),n_poses=10,overwrite=True)
        out.update(status='ok',vina=float(energies[0,0]),energies=energies.tolist())
        if calibration:
            ref = next(m for m in Chem.SDMolSupplier(task['reference_sdf'], removeHs=True) if m is not None)
            exported = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule((d/'poses.pdbqt').read_text(), skip_typing=True))
            poses = [m for m in exported if m is not None]
            assert len(poses) == 1, 'Unexpected number of ligand entities'
            probe = Chem.RemoveHs(poses[0])
            assert probe.GetNumHeavyAtoms() == ref.GetNumHeavyAtoms()
            rmsds = [float(rdMolAlign.CalcRMS(probe,ref,prbId=c.GetId(),maxMatches=100000,
                                             symmetrizeConjugatedTerminalGroups=True)) for c in probe.GetConformers()]
            assert len(rmsds) == len(energies)
            out.update(top1_rmsd_A=rmsds[0],top10_best_rmsd_A=min(rmsds),pose_rmsds_A=rmsds,
                       calibration_passed=rmsds[0]<=p['top1_rmsd_cutoff_A'])
            with Chem.SDWriter(str(d/'exported_poses.sdf')) as w:
                for c in probe.GetConformers():
                    w.write(probe,confId=c.GetId())
    except Exception as exc:
        out.update(status='technical_failure',error_class=type(exc).__name__,reason=str(exc)[:350])
    out.update(finished_utc=now(),elapsed_seconds=time.monotonic()-started)
    dump(d/'result.json',out)


def run_one(task, calibration):
    d = ROOT/('calibration' if calibration else 'docking')/task['job_id']
    d.mkdir(parents=True,exist_ok=True)
    if (d/'result.json').exists():
        return json.loads((d/'result.json').read_text())
    with (d/'worker.log').open('w') as log:
        started = time.monotonic()
        try:
            cmd=[sys.executable,str(Path(__file__)), 'calibration-worker' if calibration else 'docking-worker',task['job_id']]
            run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=600)
            if (d/'result.json').exists():
                return json.loads((d/'result.json').read_text())
            out=dict(**task,status='worker_fail',vina=None,returncode=run.returncode)
        except subprocess.TimeoutExpired:
            out=dict(**task,status='timeout',vina=None)
        out.update(elapsed_seconds=time.monotonic()-started,finished_utc=now())
        dump(d/'result.json',out)
        return out


def run_tasks(tasks, calibration):
    dump(ROOT/('calibration_tasks.json' if calibration else 'docking_tasks.json'),tasks)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs={pool.submit(run_one,t,calibration):t for t in tasks}
        for i, f in enumerate(as_completed(fs),1):
            r=f.result()
            print('CALIB' if calibration else 'DOCK',i,'/',len(tasks),r['chemblid'],r['status'],
                  r.get('top1_rmsd_A'),flush=True)
            dump(ROOT/('calibration_progress.json' if calibration else 'docking_progress.json'),
                 dict(completed=i,total=len(tasks),time_utc=now()))
    results=[json.loads((ROOT/('calibration' if calibration else 'docking')/t['job_id']/'result.json').read_text()) for t in tasks]
    dump(ROOT/('calibration_results.json' if calibration else 'docking_results.json'),results)
    return results


def calibrate():
    recs=json.loads((ROOT/'receptors.json').read_text())
    tasks=[dict(job_id=t,chemblid=t,canonical=r['reference_canonical'],receptor=r['receptor'],
                box_center=r['box_center'],box_size=r['box_size'],reference_sdf=str(ROOT/'receptors'/t/'reference_ligand.sdf'))
           for t,r in recs.items()]
    run_tasks(tasks,True)


def observations():
    sys.path.insert(0,str(B))
    from common import properties
    from rdkit.Chem import rdMolDescriptors
    recs=json.loads((ROOT/'receptors.json').read_text())
    cal={r['chemblid']:r for r in json.loads((ROOT/'calibration_results.json').read_text())}
    available={t for t,r in recs.items() if cal[t]['status']=='ok' and cal[t]['calibration_passed'] and r['primary_nonmetal']}
    protocol_hash=sha(ROOT/'docking_protocol.json')
    obs=[]
    for r in csv.DictReader((B/'inputs/targets.csv').open()):
        obs.append(dict(observation_id=r['chemblid']+'_seed',kind='seed',study='shared',group='seed',
                        chemblid=r['chemblid'],parent_smiles=r['seed_canonical']))
    for r in csv.DictReader((ROOT/'A_500_scoring_pairing_and_diagnostics.csv').open()):
        obs.append(dict(observation_id='A_'+r['group']+'_'+r['chemblid']+'_'+r['position_label'],
                        kind='generated',study='A',group=r['group'],chemblid=r['chemblid'],parent_smiles=r['canonical']))
    for path in sorted((B/'results').glob('*.json')):
        r=json.loads(path.read_text())
        obs.append(dict(observation_id='B_'+r['slot_id'],kind='generated',study='B',group=r['group'],
                        chemblid=r['chemblid'],parent_smiles=r['smiles'] if r['ok'] else None,
                        generation_status=r['status']))
    tasks={}
    for o in obs:
        t=o['chemblid']
        o['docking_subset_nonmetal_calibrated']=t in available
        if not o['parent_smiles']:
            o.update(docking_status='no_generated_output',job_id=None)
            continue
        try:
            component=active(o['parent_smiles'])
            o.update(active_component=component,parent_fragment_removed=component!=o['parent_smiles'])
            parentprops, props=properties(o['parent_smiles']),properties(component)
            o.update({f'parent_{k}':v for k,v in parentprops.items() if k!='canonical'})
            o.update({f'active_{k}':v for k,v in props.items() if k!='canonical'})
        except Exception as exc:
            o.update(docking_status='ambiguous_or_invalid_ligand',reason=str(exc),job_id=None)
            continue
        if t not in available:
            o.update(docking_status='excluded_receptor_or_calibration',job_id=None)
            continue
        r=recs[t]
        key=json.dumps([t,component,r['pdbqt_sha256'],r['box_center'],r['box_size'],protocol_hash],sort_keys=True)
        job=hashlib.sha256(key.encode()).hexdigest()[:24]
        o.update(job_id=job,docking_status='pending')
        tasks[job]=dict(job_id=job,chemblid=t,canonical=component,receptor=r['receptor'],
                       receptor_sha256=r['pdbqt_sha256'],protocol_sha256=protocol_hash,
                       box_center=r['box_center'],box_size=r['box_size'])
    assert len(obs)==900 and len({o['observation_id'] for o in obs})==900
    dump(ROOT/'observations.json',obs)
    dump(ROOT/'docking_tasks.json',list(tasks.values()))
    dump(ROOT/'calibrated_scope.json',dict(created_utc=now(),nonmetal_top1_pass_targets=sorted(available),
          primary_subset_rule='matching deposited pocket, complete CCD graph, top1 in-place heavy RMSD<=2A, no catalytic metal',
          unique_docking_jobs=len(tasks),planned_observations=900,old500=500,new375=375,seeds25=25,
          all25_generation_property_records_retained=True))
    print('LOCKED DOCKING SCOPE',len(available),'targets',len(tasks),'unique jobs',flush=True)


def dock():
    tasks=json.loads((ROOT/'docking_tasks.json').read_text())
    recs=json.loads((ROOT/'receptors.json').read_text())
    for r in recs.values():
        assert sha(r['receptor']) == r['pdbqt_sha256']
    results=run_tasks(tasks,False)
    mapping={r['job_id']:r for r in results}
    obs=json.loads((ROOT/'observations.json').read_text())
    for o in obs:
        result=mapping.get(o['job_id'])
        if result:
            o.update(docking_status=result['status'],vina=result['vina'],mmff_status=result.get('mmff_status'))
        else:
            o['vina']=None
    import pandas as pd
    pd.DataFrame(obs).to_csv(ROOT/'docking_scores.csv',index=False)
    verify()


def verify():
    expected=json.loads((ROOT/'protected_before.json').read_text())
    failures=[p for p,h in expected.items() if sha(p)!=h]
    dump(ROOT/'protected_after_verification.json',dict(checked=len(expected),changed=failures,
         verified_utc=now(),new_api_calls=0,manuscript_changed=bool(failures)))
    assert not failures


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['prepare','calibrate','observations','dock','verify','calibration-worker','docking-worker'])
    ap.add_argument('job',nargs='?')
    args=ap.parse_args()
    if args.phase.endswith('-worker'):
        calibration=args.phase.startswith('calibration')
        tasks=json.loads((ROOT/('calibration_tasks.json' if calibration else 'docking_tasks.json')).read_text())
        worker(next(t for t in tasks if t['job_id']==args.job),calibration)
    else:
        globals()[args.phase]()
