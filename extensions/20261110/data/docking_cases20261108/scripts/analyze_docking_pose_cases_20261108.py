"""Analyze saved top poses with RDKit/Meeko and PLIP; no docking or model calls.
Run: python analyze_docking_pose_cases_20261108.py --root /path/to/case_package
Dependencies: RDKit, Meeko, SciPy, PLIP 3.0.1, Open Babel, lxml.
"""
from pathlib import Path
import argparse,json,hashlib,sys,math,csv,shutil
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2)+'\n')
def native(v):
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,(str,int,float,bool)) or v is None:return v
 if isinstance(v,(tuple,list,np.ndarray)):return [native(x) for x in v]
 return str(v)

def main(root):
 # Optional isolated tools directory is not part of the experiment environment.
 if (root/'runtime').exists():sys.path.insert(0,str(root/'runtime'))
 from rdkit import Chem,RDLogger
 from rdkit.Chem import QED,Descriptors,rdMolDescriptors
 from meeko import PDBQTMolecule,RDKitMolCreate
 from scipy.spatial import cKDTree
 from plip.structure.preparation import PDBComplex
 from plip.exchange.report import BindingSiteReport
 from plip.basic import config
 import importlib.metadata
 RDLogger.DisableLog('rdApp.*')
 inputs=root/'inputs';out=root/'analysis';out.mkdir(exist_ok=True)
 manifest=json.loads((inputs/'input_manifest.json').read_text())
 assert all(sha(inputs/n)==v['sha256'] for n,v in manifest.items())
 results=[];interactions=[];checks=[]
 # PLIP default geometry and protonation; record actual settings, not assumed thresholds.
 settings={k:native(v) for k,v in vars(config).items() if k.isupper() and isinstance(v,(str,int,float,bool,tuple,list))}
 config.MAXTHREADS=1
 for condition in ['ring','tpsa']:
  case=inputs/condition;meta=json.loads((case/'case.json').read_text());d=out/condition;d.mkdir(exist_ok=True)
  protein=[];protein_xyz=[]
  for line in (case/'receptor_clean.pdb').read_text().splitlines():
   if line.startswith(('ATOM  ','HETATM')):
    element=line[76:78].strip() or line[12:16].strip()[0]
    if element not in ('H','D'):
     protein_xyz.append([float(line[30:38]),float(line[38:46]),float(line[46:54])]);protein.append(line)
  protein_xyz=np.array(protein_xyz);tree=cKDTree(protein_xyz)
  receptor_qt=[]
  for line in (case/'receptor.pdbqt').read_text().splitlines():
   if line.startswith(('ATOM  ','HETATM')) and line.split()[-1] not in ('H','HD','HS'):
    receptor_qt.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
  nearest=tree.query(np.asarray(receptor_qt))[0]
  assert float(nearest.max())<.003,'Scored receptor and clean receptor coordinates disagree.'
  case_result=dict(condition=condition,target=meta['target'],pdb=meta['receptor']['pdb_id'],roles={},
      selection_target_median=meta['target_median_delta_vina'],selected_target_mean=meta['target_mean_delta_vina'])
  for role in ['seed','output']:
   source=case/role;dest=d/role;dest.mkdir(exist_ok=True)
   task=json.loads((source/'task.json').read_text());score=json.loads((source/'result.json').read_text())
   assert sha(case/'receptor.pdbqt')==task['receptor']['sha256']
   poses=PDBQTMolecule.from_file(str(source/'poses.pdbqt'),skip_typing=True)
   mols=RDKitMolCreate.from_pdbqt_mol(poses);assert len(mols)==1 and mols[0] is not None
   allposes=mols[0];mol=Chem.Mol(allposes);conf=Chem.Conformer(allposes.GetConformer(0));mol.RemoveAllConformers();mol.AddConformer(conf)
   mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
   expected=Chem.MolFromSmiles(task['canonical']);canonical=Chem.MolToSmiles(mol,isomericSmiles=True)
   assert canonical==Chem.MolToSmiles(expected,isomericSmiles=True),(condition,role,canonical,task['canonical'])
   # Verify that reconstructed heavy atoms are exactly the first scored pose.
   first=[]
   for line in (source/'poses.pdbqt').read_text().splitlines():
    if line.startswith('ENDMDL'):break
    if line.startswith(('ATOM  ','HETATM')) and line.split()[-1] not in ('H','HD','HS'):
     first.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
   xyz=np.asarray(mol.GetConformer().GetPositions());dist=cKDTree(np.asarray(first)).query(xyz)[0]
   assert len(first)==len(xyz) and dist.max()<.003
   first_score=next(float(line.split()[3]) for line in (source/'poses.pdbqt').read_text().splitlines() if line.startswith('REMARK VINA RESULT:'))
   assert abs(first_score-float(score['vina']))<1e-6
   for atom in mol.GetAtoms():
    info=Chem.AtomPDBResidueInfo();info.SetName((atom.GetSymbol()+str(atom.GetIdx()+1)).rjust(4));info.SetResidueName('LIG');info.SetResidueNumber(999);info.SetChainId('Z');info.SetIsHeteroAtom(True);info.SetSerialNumber(atom.GetIdx()+1);atom.SetMonomerInfo(info)
   ligandblock=Chem.MolToPDBBlock(mol)
   (dest/'top_pose.pdb').write_text(ligandblock)
   writer=Chem.SDWriter(str(dest/'top_pose.sdf'));writer.write(mol);writer.close()
   # Protein and ligand remain in the scored receptor frame; never align the ligands to each other.
   lines=[x for x in (case/'receptor_clean.pdb').read_text().splitlines() if x.startswith(('ATOM  ','HETATM','TER'))]
   maxserial=max(int(x[6:11]) for x in lines if x.startswith(('ATOM  ','HETATM')))
   ligandlines=[]
   for x in ligandblock.splitlines():
    if x.startswith(('ATOM  ','HETATM')):ligandlines.append(x[:6]+f'{int(x[6:11])+maxserial+1:5d}'+x[11:])
    elif x.startswith('CONECT'):
     indices=[int(x[i:i+5]) for i in range(6,len(x),5) if x[i:i+5].strip()]
     ligandlines.append('CONECT'+''.join(f'{i+maxserial+1:5d}' for i in indices))
   complexfile=dest/'complex.pdb';complexfile.write_text('\n'.join(lines+ligandlines+['END'])+'\n')
   # Freeze polar-H coordinates once, then reuse them for deterministic contact annotation.
   # Heavy-atom poses and all docking scores remain unchanged.
   frozen=dest/'frozen_annotation_complex.pdb'
   if not frozen.exists():
    prior=dest/'plip/complex_protonated.pdb'
    if prior.exists():shutil.copy2(prior,frozen)
    else:
     initial=dest/'initial_hydrogens';initial.mkdir(exist_ok=True)
     config.NOHYDRO=False;config.OUTPATH=str(initial)
     first_model=PDBComplex();first_model.output_path=str(initial);first_model.load_pdb(str(complexfile))
     shutil.copy2(initial/'complex_protonated.pdb',frozen)
   frozen_heavy=[]
   for line in frozen.read_text().splitlines():
    if line.startswith(('ATOM  ','HETATM')) and line[76:78].strip() not in ('H','D'):
     frozen_heavy.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
   assert cKDTree(np.vstack([protein_xyz,xyz])).query(np.asarray(frozen_heavy))[0].max()<.003
   config.NOHYDRO=True;config.OUTPATH=str(dest/'plip_frozen')
   model=PDBComplex();model.output_path=str(dest/'plip_frozen');model.load_pdb(str(frozen));model.analyze()
   sites=[(k,s) for k,s in model.interaction_sets.items() if k.startswith('LIG:Z:999')]
   assert len(sites)==1,list(model.interaction_sets)
   key,site=sites[0];report=BindingSiteReport(site)
   # The reporting interface exposes typed, atom-level lists with feature names.
   tables={}
   for attr in vars(report):
    if attr.endswith('_features'):
     base=attr[:-9];features=list(getattr(report,attr));info=getattr(report,base+'_info',None)
     if info is not None:
      table=[{str(k):native(v) for k,v in zip(features,record)} for record in info];tables[base]=table
      for record in table:interactions.append(dict(condition=condition,role=role,type=base,**record))
   dump(dest/'interactions.json',tables)
   rawtypes={k:type(v).__name__ for k,v in vars(report).items()}
   dump(dest/'report_interface.json',rawtypes)
   properties=dict(QED=float(QED.qed(mol)),RC=int(rdMolDescriptors.CalcNumRings(mol)),TPSA=float(rdMolDescriptors.CalcTPSA(mol)),MW=float(Descriptors.MolWt(mol)))
   native_meta=meta[role];target_QED=float(native_meta['QED']);assert abs(properties['QED']-target_QED)<1e-9
   nearest_protein=tree.query(xyz)[0]
   role_result=dict(job_id=task['job_id'],canonical=canonical,vina=float(score['vina']),**properties,
    coordinates_match_first_pose=True,heavy_atoms=len(xyz),input_pose_count=allposes.GetNumConformers(),
    min_protein_ligand_heavy_distance=float(nearest_protein.min()),interaction_counts={k:len(v) for k,v in tables.items()},
    top_pose_sdf_sha256=sha(dest/'top_pose.sdf'),complex_sha256=sha(complexfile),frozen_hydrogen_complex_sha256=sha(frozen))
   case_result['roles'][role]=role_result
   checks.append(dict(condition=condition,role=role,canonical_identity=True,score_matches=True,first_pose_heavy_coords_max_error=float(dist.max()),
    receptor_heavy_coords_max_error=float(nearest.max()),input_pose_sha256=sha(source/'poses.pdbqt')))
  case_result['delta_QED']=case_result['roles']['output']['QED']-case_result['roles']['seed']['QED']
  case_result['delta_vina']=case_result['roles']['output']['vina']-case_result['roles']['seed']['vina']
  results.append(case_result)
 dump(out/'cases.json',results);dump(out/'all_interactions.json',interactions)
 settings['NOHYDRO']=True
 dump(out/'verification.json',dict(status='passed',cases=2,structures=4,new_docking_calls=0,new_generation_calls=0,checks=checks,annotation_hydrogens_frozen=True,
  tool_versions={n:importlib.metadata.version(n) for n in ['rdkit','meeko','plip','openbabel-wheel','pymol-open-source']},plip_settings=settings))
 fields=list(dict.fromkeys(k for r in interactions for k in r))
 with (out/'all_interactions.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(interactions)
 print(json.dumps(dict(status='passed',cases=results,interaction_rows=len(interactions))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/ProLiGen_docking_cases_20261108'));a=p.parse_args();main(a.root.resolve())
