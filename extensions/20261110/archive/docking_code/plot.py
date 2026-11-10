"""Native PyMOL views plus data-derived contact fingerprints for two saved poses."""
from pathlib import Path
import json,sys,csv,hashlib,argparse
import numpy as np

def main(root,reuse_views=False):
 if (root/'runtime').exists():sys.path.insert(0,str(root/'runtime'))
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from matplotlib.lines import Line2D
 import pymol2
 from PIL import Image
 a=root/'analysis';f=root/'figures';f.mkdir(exist_ok=True)
 cases=json.loads((a/'cases.json').read_text());allrows=json.loads((a/'all_interactions.json').read_text())
 blue='#286A9D';orange='#D98A36';palette={'seed':blue,'output':orange}
 # Display rule is common to both structures/cases. Preserve default annotations separately.
 plotted=[];profiles=[]
 for case in cases:
  cond=case['condition'];part=[x for x in allrows if x['condition']==cond]
  for r in part:
   displaytype=None
   if r['type']=='hydrophobic':displaytype='Hydrophobic'
   elif r['type']=='pistacking':displaytype='Aromatic'
   elif r['type']=='hbond' and float(r['DIST_D-A'])<=3.5 and float(r['DON_ANGLE'])>=120:displaytype='Short H-bond'
   if displaytype:
    profiles.append(dict(condition=cond,role=r['role'],kind=displaytype,reschain=r['RESCHAIN'],residue=r['RESTYPE'],resnr=int(r['RESNR'])))
  if reuse_views and (f/(cond+'_pose.png')).exists():continue
  # One shared receptor frame, no fitting of one ligand onto the other.
  with pymol2.PyMOL() as pm:
   cmd=pm.cmd;cmd.set('max_threads',2);cmd.reinitialize()
   cmd.load(str(root/'inputs'/cond/'receptor_clean.pdb'),'protein')
   cmd.load(str(a/cond/'seed/top_pose.sdf'),'seed')
   cmd.load(str(a/cond/'output/top_pose.sdf'),'edited')
   cmd.hide('everything','all');cmd.remove('hydro')
   cmd.select('ligands','seed or edited')
   cmd.select('pocket','byres (protein within 4.5 of ligands)')
   cmd.select('nearby','byres (protein within 10 of ligands)')
   cmd.set_color('seed_carbon',[.157,.416,.616]);cmd.set_color('edit_carbon',[.851,.541,.212]);cmd.set_color('pocket_grey',[.67,.70,.73])
   cmd.color('pocket_grey','protein');cmd.show('cartoon','nearby');cmd.set('cartoon_transparency',.88,'protein')
   cmd.set('cartoon_oval_width',.12);cmd.set('cartoon_oval_length',.70)
   cmd.show('sticks','pocket');cmd.set('stick_radius',.09,'protein')
   cmd.show('sticks','ligands');cmd.set('stick_radius',.20,'ligands')
   cmd.color('seed_carbon','seed and elem C');cmd.color('edit_carbon','edited and elem C')
   for selection in ['seed','edited','pocket']:
    for elem,color in [('N','blue'),('O','red'),('S','yellow'),('F','palegreen'),('Br','brown')]:cmd.color(color,selection+' and elem '+elem)
   cmd.set('stick_quality',16);cmd.set('sphere_quality',2)
   labels=set()
   for j,r in enumerate(part):
    short=r['type']=='hbond' and float(r['DIST_D-A'])<=3.5 and float(r['DON_ANGLE'])>=120
    if short or r['type']=='pistacking':
     labels.add((r['RESCHAIN'],int(r['RESNR']),r['RESTYPE']))
    if short:
     p1=f'ligcontact{j}';p2=f'protcontact{j}';name=f'hbond{j}'
     cmd.pseudoatom(p1,pos=r['LIGCOO']);cmd.pseudoatom(p2,pos=r['PROTCOO']);cmd.distance(name,p1,p2)
     cmd.hide('labels',name);cmd.set('dash_color','seed_carbon' if r['role']=='seed' else 'edit_carbon',name)
     cmd.set('dash_width',2.5,name);cmd.set('dash_gap',.25,name);cmd.set('dash_radius',.04,name)
     cmd.hide('everything',p1+' or '+p2)
   for li,(chain,number,res) in enumerate(sorted(labels)):
    contacts=[r for r in part if r['RESCHAIN']==chain and int(r['RESNR'])==number and (r['type']=='hbond' or r['type']=='pistacking')]
    label_pos=np.mean([r['PROTCOO'] for r in contacts],axis=0).tolist()
    obj='reslabel'+str(li);cmd.pseudoatom(obj,pos=label_pos);cmd.hide('everything',obj)
    cmd.label(obj,'"'+res.title()+str(number)+'"');cmd.show('labels',obj)
   cmd.set('label_font_id',7);cmd.set('label_size',22);cmd.set('label_color','black');cmd.set('label_outline_color','white')
   cmd.set('label_position',[.8,1.0,.4]);cmd.bg_color('white')
   cmd.set('orthoscopic',1);cmd.set('depth_cue',0);cmd.set('ray_shadows',0);cmd.set('specular',.20)
   cmd.set('ambient',.55);cmd.set('direct',.65);cmd.set('antialias',2)
   cmd.orient('ligands');cmd.rotate('y',20);cmd.rotate('x',15);cmd.zoom('ligands',6.5)
   cmd.save(str(f/(cond+'_scene.pse')))
   cmd.png(str(f/(cond+'_pose.png')),width=1900,height=1400,dpi=300,ray=1)
   (f/(cond+'_camera.json')).write_text(json.dumps(list(cmd.get_view()),indent=2)+'\n')
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42})
 fig=plt.figure(figsize=(10.8,10.3));gs=fig.add_gridspec(2,2,width_ratios=[1.35,1.],hspace=.48,wspace=.28)
 for i,case in enumerate(cases):
  cond=case['condition'];ax=fig.add_subplot(gs[i,0]);image=np.asarray(Image.open(f/(cond+'_pose.png')).convert('RGB'))
  # Crop background only; never alter molecular coordinates, labels or representations.
  nonwhite=np.min(image,axis=2)<242;yy,xx=np.nonzero(nonwhite);pad=55
  crop=image[max(0,yy.min()-pad):min(image.shape[0],yy.max()+pad),max(0,xx.min()-pad):min(image.shape[1],xx.max()+pad)]
  ax.imshow(crop);ax.axis('off')
  title='Ring editing' if cond=='ring' else 'TPSA editing'
  ax.set_title(f"({'a' if i==0 else 'c'}) {title}: {case['pdb']}",loc='left',fontsize=14,fontweight='bold',pad=10)
  seed=case['roles']['seed'];edit=case['roles']['output']
  metric=f"Ring count {seed['RC']} → {edit['RC']}" if cond=='ring' else f"TPSA {seed['TPSA']:.2f} → {edit['TPSA']:.2f} Å²"
  ax.text(.00,-.07,metric+f"\nΔVina {case['delta_vina']:+.3f} kcal/mol",transform=ax.transAxes,fontsize=13,va='top')
  ax2=fig.add_subplot(gs[i,1]);part=[x for x in profiles if x['condition']==cond]
  residues=sorted({(r['reschain'],r['resnr'],r['residue']) for r in part},key=lambda t:(t[0],t[1]))
  kinds=['Hydrophobic','Short H-bond','Aromatic'];resmap={t:j for j,t in enumerate(residues)}
  for j in range(len(residues)):
   if j%2==0:ax2.axhspan(j-.5,j+.5,color='#F3F5F7',zorder=0)
  unique=sorted({(x['role'],x['kind'],x['reschain'],x['resnr'],x['residue']) for x in part})
  for role,kind,chain,number,res in unique:
   x=kinds.index(kind)+(-.12 if role=='seed' else .12);y=resmap[(chain,number,res)]
   ax2.scatter(x,y,s=48,color=palette[role],edgecolor='white',linewidth=.6,zorder=3)
   plotted.append(dict(condition=cond,role=role,contact_type=kind,chain=chain,resnr=number,residue=res,x=x,y=y))
  ax2.set_yticks(range(len(residues)),[f'{t[2].title()}{t[1]} ({t[0]})' for t in residues],fontsize=13)
  ax2.set_xticks(range(3),['Hydrophobic','Short\nH-bond','Aromatic'],fontsize=13)
  ax2.tick_params(axis='both',length=0,pad=5);ax2.set_xlim(-.5,2.5);ax2.set_ylim(len(residues)-.5,-.5)
  for spine in ax2.spines.values():spine.set_visible(False)
  ax2.set_title(f"({'b' if i==0 else 'd'}) Residue contacts",loc='left',fontsize=14,fontweight='bold',pad=10)
 handles=[Line2D([0],[0],color=blue,lw=3,label='Seed'),Line2D([0],[0],color=orange,lw=3,label='Edited output')]
 fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.993),frameon=False,ncol=2,fontsize=14)
 fig.subplots_adjust(left=.035,right=.985,top=.90,bottom=.13)
 fig.savefig(f/'docking_pose_cases.png',dpi=300,facecolor='white')
 fig.savefig(f/'docking_pose_cases.pdf',facecolor='white');fig.savefig(f/'docking_pose_cases.svg',facecolor='white');plt.close(fig)
 with (a/'figure_plotted_contacts.csv').open('w',newline='') as stream:
  w=csv.DictWriter(stream,fieldnames=list(plotted[0]));w.writeheader();w.writerows(plotted)
 (a/'figure_definition.json').write_text(json.dumps(dict(structures='Native first poses, same receptor frame, no ligand fitting',
  contact_display='PLIP hydrophobic/aromatic contacts; H-bonds additionally require donor-acceptor <=3.5 Angstrom and donor angle >=120 degrees',
  full_default_annotations='all_interactions.csv',plotted_points=len(plotted),seed_color=blue,output_color=orange,
  source_sha256={n:hashlib.sha256((a/n).read_bytes()).hexdigest() for n in ['cases.json','all_interactions.json']},
  plot_sha256=hashlib.sha256((f/'docking_pose_cases.png').read_bytes()).hexdigest()),indent=2)+'\n')
 print(json.dumps(dict(status='complete',cases=2,plotted_points=len(plotted),new_docking_calls=0,new_generation_calls=0)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('/root/ProLiGen_docking_cases_20261108'));p.add_argument('--reuse-views',action='store_true');args=p.parse_args();main(args.root.resolve(),reuse_views=args.reuse_views)
