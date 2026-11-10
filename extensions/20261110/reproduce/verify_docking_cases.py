"""Offline scientific-data and figure-source checks. Python 3.10+; standard library only.
This entry point does not import PLIP, PyMOL or generation/docking engines.
"""
from pathlib import Path
import csv,json,hashlib,math,statistics
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/docking_cases20261108'
def rows(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def close(a,b,tol=1e-9):assert abs(float(a)-float(b))<tol,(a,b)
def xyz(line):return [float(line[30:38]),float(line[38:46]),float(line[46:54])]
def dist(a,b):return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
def main():
 manifest=json.loads((D/'manifest.json').read_text(encoding='utf-8'))
 for name,h in manifest['files'].items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==h,name
 cases=json.loads((D/'analysis/cases.json').read_text());annotations=json.loads((D/'analysis/all_interactions.json').read_text())
 assert len(cases)==2 and len(annotations)==44
 expected=set()
 for r in annotations:
  kind=None
  if r['type']=='hydrophobic':kind='Hydrophobic'
  elif r['type']=='pistacking':kind='Aromatic'
  elif r['type']=='hbond':
   close(dist(r['LIGCOO'],r['PROTCOO']),r['DIST_D-A'],tol=.011)
   if float(r['DIST_D-A'])<=3.5 and float(r['DON_ANGLE'])>=120:kind='Short H-bond'
  if kind:expected.add((r['condition'],r['role'],kind,r['RESCHAIN'],int(r['RESNR']),r['RESTYPE']))
 points=rows(D/'analysis/figure_plotted_contacts.csv')
 actual={(r['condition'],r['role'],r['contact_type'],r['chain'],int(r['resnr']),r['residue']) for r in points}
 assert actual==expected and len(actual)==len(points)
 definition=json.loads((D/'analysis/figure_definition.json').read_text())
 assert hashlib.sha256((D/'figures/docking_pose_cases.png').read_bytes()).hexdigest()==definition['plot_sha256']
 assert hashlib.sha256((D/'figures/docking_pose_cases.pdf').read_bytes()).hexdigest()==hashlib.sha256((ROOT/'figures/docking_pose_cases.pdf').read_bytes()).hexdigest()
 for condition in ['ring','tpsa']:
  part=[r for r in points if r['condition']==condition]
  residues=sorted({(r['chain'],int(r['resnr']),r['residue']) for r in part},key=lambda x:(x[0],x[1]))
  for r in part:
   close(r['x'],['Hydrophobic','Short H-bond','Aromatic'].index(r['contact_type'])+(-.12 if r['role']=='seed' else .12))
   close(r['y'],residues.index((r['chain'],int(r['resnr']),r['residue'])))
 source=ROOT/'data/mechanism_docking20261102'
 # Supporting archive includes the two selection-source CSVs under this dataset as well.
 taskfile=source/'docking_target_means231.csv';obsfile=source/'docking_observations759.csv'
 if not taskfile.exists():taskfile=D/'selection_sources/docking_target_means231.csv';obsfile=D/'selection_sources/docking_observations759.csv'
 tasks=rows(taskfile);obs=rows(obsfile)
 for c in cases:
  cond=c['condition'];meta=json.loads((D/'inputs'/cond/'case.json').read_text())
  ts=[x for x in tasks if x['condition']==cond and x['group']=='G' and x['kind']=='generated'];assert len(ts)==11
  median=statistics.median(float(x['delta_vina']) for x in ts)
  selected=min(ts,key=lambda x:(abs(float(x['delta_vina'])-median),x['chemblid']));assert selected['chemblid']==c['target']
  available=[x for x in obs if x['condition']==cond and x['group']=='G' and x['chemblid']==c['target'] and x['docking_ok']=='1']
  selectedobs=min(available,key=lambda x:(abs(float(x['delta_vina'])-float(selected['delta_vina'])),x['slot_id']))
  assert selectedobs['slot_id']==meta['output']['slot_id']
  for role in ['seed','output']:
   original=D/'inputs'/cond/role;raw=(original/'poses.pdbqt').read_text()
   score=next(float(l.split()[3]) for l in raw.splitlines() if l.startswith('REMARK VINA RESULT:'))
   result=json.loads((original/'result.json').read_text());close(score,result['vina']);close(score,c['roles'][role]['vina'])
   native=[]
   for l in raw.splitlines():
    if l.startswith('ENDMDL'):break
    if l.startswith(('ATOM  ','HETATM')) and l.split()[-1] not in ('H','HD','HS'):native.append(xyz(l))
   reconstructed=[xyz(l) for l in (D/'analysis'/cond/role/'top_pose.pdb').read_text().splitlines() if l.startswith(('ATOM  ','HETATM'))]
   assert len(native)==len(reconstructed)
   assert max(min(dist(v,w) for w in native) for v in reconstructed)<.003
  close(c['roles']['output']['vina']-c['roles']['seed']['vina'],c['delta_vina'])
  close(c['roles']['output']['QED']-c['roles']['seed']['QED'],c['delta_QED'])
  close(c['delta_vina'],meta['output']['delta_vina']);close(c['delta_QED'],meta['output']['delta_QED'])
 result=dict(status='passed',cases=2,top_poses=4,default_contact_records=44,plotted_contact_marks=len(points),
  case_selection_reproduced=True,first_pose_scores_and_coords_verified=True,API_calls=0,new_docking_calls=0)
 print(json.dumps(result,indent=2));return result
if __name__=='__main__':main()
