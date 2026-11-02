"""Recreate Figure 6 from verified observation-linked task/group means."""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mechanism_docking import main as verify
P=Path(__file__).resolve().parents[1];D=P/'data/mechanism_docking20261102'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    verify()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,
        'axes.linewidth':.7,'xtick.labelsize':10,'ytick.labelsize':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'T':'#707B84','M':'#2874A6','G':'#8056A3','R':'#CC7B2B','LIB_NN1':'#406B49','LIB_QED3':'#B14C59','LIB_QED6':'#333333'}
    order=list(colors);labels={'LIB_NN1':'NN-1','LIB_QED3':'QED-3','LIB_QED6':'QED-6'}
    tasks=rows(D/'figure6_target_points132.csv');means=rows(D/'figure6_means21.csv');coords=[]
    fig,axes=plt.subplots(1,3,figsize=(9.35,3.65),sharex=True,sharey=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.28,top=.86,wspace=.16)
    for ai,(axis,c) in enumerate(zip(axes,['ring','logp','tpsa'])):
        for g in order:
            if g in ['T','M','G','R']:
                part=[r for r in tasks if r['condition']==c and r['group']==g]
                axis.scatter([float(r['delta_vina']) for r in part],[float(r['delta_QED']) for r in part],s=10,color=colors[g],alpha=.40,linewidths=0,zorder=2)
                for r in part:coords.append(dict(condition=c,group=g,kind='task',chemblid=r['chemblid'],x=r['delta_vina'],y=r['delta_QED']))
            r=next(r for r in means if r['condition']==c and r['group']==g)
            x,y,lo,hi=[float(r[k]) for k in ['delta_vina','delta_QED','delta_vina_ci_low','delta_vina_ci_high']]
            direct=g.startswith('LIB_')
            axis.errorbar(x,y,xerr=np.array([[x-lo],[hi-x]]),fmt='s' if direct else 'D',mfc='white' if direct else colors[g],mec=colors[g],mew=1,
                ms=5.5,color=colors[g],lw=1,capsize=2,zorder=5)
            coords.append(dict(condition=c,group=g,kind='mean',x=x,y=y,ci_low=lo,ci_high=hi))
        axis.axhline(0,color='#b7b7b7',lw=.7,zorder=0);axis.axvline(0,color='#b7b7b7',lw=.7,zorder=0)
        axis.set_title('('+chr(97+ai)+') '+{'ring':'Ring','logp':'LogP','tpsa':'TPSA'}[c],fontsize=11,pad=8)
        axis.set_xlim(-1.8,5.1);axis.set_ylim(-.30,.60);axis.set_xticks([-1,0,2,4]);axis.set_yticks([-.2,0,.2,.4,.6])
    fig.text(.52,.12,'Vina change: output - seed (kcal/mol)',ha='center',fontsize=11)
    fig.text(.013,.57,'QED change: output - seed',rotation=90,va='center',fontsize=11)
    handles=[Line2D([0],[0],marker='s' if g.startswith('LIB_') else 'D',linestyle='none',markerfacecolor='white' if g.startswith('LIB_') else colors[g],markeredgecolor=colors[g],markersize=5.5,label=labels.get(g,g)) for g in order]
    fig.legend(handles=handles,ncol=7,loc='lower center',bbox_to_anchor=(.51,-.015),frameon=False,columnspacing=1.25,handletextpad=.35,fontsize=10)
    for suffix in ['pdf','png','svg']:fig.savefig(P/'figures'/('fig5_docking_tradeoffs.'+suffix),dpi=300,bbox_inches='tight',pad_inches=.07,facecolor='white')
    plt.close(fig)
    with (D/'figure6_plotted_coordinates.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in coords for k in r)));w.writeheader();w.writerows(coords)
    assert len(coords)==153
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'figure6_target_points132.csv',D/'figure6_means21.csv',D/'figure6_plotted_coordinates.csv',P/'figures/fig5_docking_tradeoffs.pdf']}
    (D/'figure6_source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    print('Figure 6: 132 generated task points, 21 means and 21 intervals; coordinates archived.')
if __name__=='__main__':main()
