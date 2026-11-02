"""Recreate Figure 5 from the independently verified frozen observations.
Requires Python 3.10+, NumPy and Matplotlib. No model calls.
"""
from pathlib import Path
import csv,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extended_comparison import main as verify
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'data/updated20261102'
def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    verify()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelsize':10,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.7,'xtick.labelsize':9,'ytick.labelsize':9})
    colors={'ring':'#276A9F','logp':'#C47932','tpsa':'#6B6997'}
    names={'ring':'Ring','logp':'LogP','tpsa':'TPSA'}
    conditions=['ring','logp','tpsa'];coords=[]
    fig,(a,b)=plt.subplots(1,2,figsize=(9.3,3.65),gridspec_kw={'width_ratios':[1,1.5]})
    fig.subplots_adjust(left=.08,right=.90,bottom=.18,top=.87,wspace=.48)
    points=rows(D/'figure5_target_differences75.csv')
    for i,c in enumerate(conditions):
        part=sorted([r for r in points if r['condition']==c],key=lambda r:r['chemblid'])
        jitter=np.random.default_rng(42+i).uniform(-.16,.16,len(part))
        vals=np.array([float(r['difference']) for r in part])
        a.scatter(i+jitter,vals,s=17,color=colors[c],alpha=.72,linewidths=0,zorder=3)
        a.scatter(i,vals.mean(),marker='D',s=51,color=colors[c],edgecolor='white',linewidth=.8,zorder=4)
        for r,x in zip(part,i+jitter):coords.append(dict(panel='a',kind='task',condition=c,chemblid=r['chemblid'],comparison='G-M',x=float(x),y=float(r['difference'])))
        coords.append(dict(panel='a',kind='mean',condition=c,comparison='G-M',x=i,y=float(vals.mean())))
    a.axhline(0,color='#666666',lw=.8,ls='--',zorder=1)
    a.set_xticks(range(3),[names[c] for c in conditions]);a.set_xlim(-.45,2.45)
    a.set_ylabel('QED-yield difference (G - M)');a.set_title('(a) Paired task differences',loc='left',fontsize=11,pad=13)
    a.grid(axis='y',color='#ededed',lw=.6,zorder=0)
    contrasts=rows(D/'figure5_comparisons9.csv');ticks=[];labels=[]
    for ci,c in enumerate(conditions):
        for j,comparison in enumerate(['M-R','G-R','G-M']):
            r=next(r for r in contrasts if r['condition']==c and r['comparison']==comparison)
            y=8-(3*ci+j);x,lo,hi,p=[float(r[k]) for k in ['difference','ci_low','ci_high','p_holm']]
            b.errorbar(x,y,xerr=np.array([[x-lo],[hi-x]]),fmt='o',ms=4.8,color=colors[c],capsize=2.8,lw=1.2,zorder=3)
            ticks.append(y);labels.append(names[c]+'  '+comparison.replace('-',' - '))
            b.text(1.035,y,f'{p:.3f}',transform=b.get_yaxis_transform(),ha='left',va='center',fontsize=9,color='#333333',clip_on=False)
            coords.append(dict(panel='b',kind='contrast',condition=c,comparison=comparison,x=x,y=y,ci_low=lo,ci_high=hi,p_holm=p))
    b.set_yticks(ticks,labels);b.set_ylim(-.65,8.65);b.set_xlim(-.072,.086)
    b.set_xticks([-.06,-.03,0,.03,.06]);b.set_xticklabels(['-0.06','-0.03','0','0.03','0.06'])
    b.axvline(0,color='#666666',lw=.8,ls='--',zorder=1)
    for y in [2.5,5.5]:b.axhline(y,color='#e6e6e6',lw=.7,zorder=0)
    b.set_xlabel('Mean QED-yield difference');b.set_title('(b) Ranking comparisons',loc='left',fontsize=11,pad=13)
    b.text(1.035,1.055,'Holm p',transform=b.transAxes,ha='left',fontsize=9)
    for suffix in ['pdf','png','svg']:
        fig.savefig(ROOT/'figures'/('fig4_downstream.'+suffix),dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.06)
    plt.close(fig)
    keys=list(dict.fromkeys(k for r in coords for k in r))
    with (D/'figure5_plotted_coordinates.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(coords)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [D/'figure5_target_differences75.csv',D/'figure5_comparisons9.csv',D/'figure5_plotted_coordinates.csv',ROOT/'figures/fig4_downstream.pdf']}
    (D/'figure5_source_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    assert len(coords)==87
    print('Figure 5: 75 task points, 3 means, 9 intervals; coordinates archived.')
if __name__=='__main__':main()
