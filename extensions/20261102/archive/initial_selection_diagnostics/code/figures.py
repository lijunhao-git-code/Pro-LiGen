"""Draw scientific plots exclusively from exported replay tables."""
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from retained_methods import interval
GROUPS=['T','M','G'];COLORS={'T':'#667085','M':'#087F8C','G':'#D87A22'}
NAMES={'T':'No examples (T)','M':'Morgan examples (M)','G':'GIN examples (G)'}
POLICIES=['first_eligible','random_unique_expected','best_QED']
LABELS=['First eligible','Uniform random: expectation','Highest QED']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})


def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    if (out/'figure_index.json').exists():raise RuntimeError('Use a fresh directory to render again')
    r=ROOT/'results';target=pd.read_csv(r/'policy_target_means.csv');summary=pd.read_csv(r/'policy_summary.csv')
    cq=pd.read_csv(r/'candidate_target_quality75.csv');grid=pd.read_csv(r/'candidate_grid2250.csv')
    points=[];index=[]
    def save(fig,name,sources,note):
        for ext in ['png','svg']:fig.savefig(out/(name+'.'+ext),dpi=240,bbox_inches='tight',pad_inches=.13)
        plt.close(fig)
        index.append(dict(figure=name,sources={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},note=note))
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.7),layout='constrained')
    for ax,metric,title,label in zip(axes,['yield_QED','tanimoto'],
        ['A  Selected QED yield','B  Selected similarity to seed'],['QED per planned run','Morgan Tanimoto']):
        for gi,g in enumerate(GROUPS):
            means=[];lows=[];highs=[];xs=np.arange(3)+(gi-1)*.04
            for pi,policy in enumerate(POLICIES):
                part=target[(target.group==g)&(target.policy==policy)&(target.k==6)].sort_values('chemblid')
                vals=part[metric].to_numpy();assert len(vals)==25
                est=interval(vals);means.append(est['difference']);lows.append(est['ci_low']);highs.append(est['ci_high'])
                points.append(dict(figure='01_selection_policies',panel=metric,group=g,policy=policy,k=6,chemblid=None,
                    x=xs[pi],y=est['difference'],ci_low=est['ci_low'],ci_high=est['ci_high'],mark='target_mean_with_CI'))
            ax.errorbar(xs,means,yerr=[np.array(means)-lows,np.array(highs)-means],color=COLORS[g],
                        marker='o',ms=6,lw=1.8,capsize=4,label=NAMES[g])
        ax.set_xticks(range(3),LABELS,rotation=12,ha='right');ax.set_xlim(-.15,2.15)
        ax.set_ylabel(label);ax.set_title(title,loc='left');ax.grid(axis='y',color='#E2E8F0',lw=.6);ax.set_axisbelow(True)
    axes[0].set_ylim(.4,.8);axes[1].set_ylim(.45,.8)
    axes[0].legend(loc='upper left',fontsize=9,frameon=False)
    save(fig,'01_selection_policies',[r/'policy_target_means.csv'],
         'Fixed saved candidates, up to six positions. Means and unadjusted95% target bootstrap intervals, n=25. Random is the exact expectation over distinct eligible structures. Tanimoto conditional on delivery.')

    fig,axes=plt.subplots(2,2,figsize=(11.3,8),layout='constrained')
    for j,(metric,title) in enumerate([('valid_QED','A  All valid proposals'),('eligible_QED','B  Eligible proposals')]):
        ax=axes[0,j]
        for i,g in enumerate(GROUPS):
            part=cq[cq.group==g].sort_values('chemblid');vals=part[metric].to_numpy();xs=i+np.linspace(-.16,.16,25)
            ax.scatter(xs,vals,s=24,c=COLORS[g],alpha=.6,linewidths=0)
            ax.scatter(i,vals.mean(),s=65,c=COLORS[g],edgecolors='black',linewidths=.6,marker='D',zorder=4)
            ax.text(i,1.02,f'{vals.mean():.3f}',ha='center',transform=ax.get_xaxis_transform(),color=COLORS[g])
            for x,(_,row) in zip(xs,part.iterrows()):
                points.append(dict(figure='02_candidate_quality',panel=metric,group=g,policy=None,k=None,
                                   chemblid=row.chemblid,x=x,y=row[metric],mark='target_mean'))
        ax.set_xticks(range(3),['No examples','Morgan','GIN']);ax.set_ylim(0,1);ax.set_xlim(-.45,2.45)
        ax.set_ylabel('Mean proposal QED within target');ax.set_title(title,loc='left',pad=25)
    for j,(flag,title) in enumerate([('valid','C  QED distribution of all valid positions'),('eligible','D  QED distribution of eligible positions')]):
        ax=axes[1,j]
        for g in GROUPS:
            vals=np.sort(grid[(grid.group==g)&(grid[flag]==True)].QED.to_numpy())
            ys=np.arange(1,len(vals)+1)/len(vals)
            ax.step(vals,ys,where='post',color=COLORS[g],lw=2,label=f'{NAMES[g]} (n={len(vals)})')
            for x,y in zip(vals,ys):
                points.append(dict(figure='02_candidate_quality',panel=flag+'_ECDF',group=g,
                    policy=None,k=None,chemblid=None,x=x,y=y,mark='proposal_position'))
        ax.set_xlim(0,1);ax.set_ylim(0,1.02);ax.set_xlabel('Proposal QED');ax.set_ylabel('Fraction of positions at or below x')
        ax.set_title(title,loc='left',fontsize=11);ax.legend(fontsize=8,loc='upper left',frameon=False)
    for ax in axes.ravel():ax.grid(color='#E2E8F0',lw=.6);ax.set_axisbelow(True)
    save(fig,'02_candidate_quality',[r/'candidate_target_quality75.csv',r/'candidate_grid2250.csv'],
         'Top: 25 target means, no final-QED selection. Bottom: descriptive proposal-position ECDF, repeats retained, invalid/missing excluded here; validity and zero-filled proposal yields reported separately. No independent-proposal inferential test.')

    fig,axes=plt.subplots(2,3,figsize=(12.1,7.3),layout='constrained')
    styles={'first_eligible':(':','o'),'random_unique_expected':('--','s'),'best_QED':('-','D')}
    for j,g in enumerate(GROUPS):
        part=summary[(summary.group==g)&(summary.policy=='best_QED')].sort_values('k')
        axes[0,j].plot(part.k,part.delivery,color=COLORS[g],lw=2,marker='o')
        for _,row in part.iterrows():
            axes[0,j].annotate(str(int(row.delivered)),(row.k,row.delivery),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
            points.append(dict(figure='03_candidate_count',panel='delivery',group=g,policy='all_same_delivery',k=row.k,
                               chemblid=None,x=row.k,y=row.delivery,mark='target_mean'))
        for policy in POLICIES:
            part=summary[(summary.group==g)&(summary.policy==policy)].sort_values('k');ls,marker=styles[policy]
            axes[1,j].plot(part.k,part.yield_QED,color=COLORS[g],linestyle=ls,marker=marker,ms=5,lw=1.8)
            for _,row in part.iterrows():points.append(dict(figure='03_candidate_count',panel='yield_QED',group=g,
                policy=policy,k=row.k,chemblid=None,x=row.k,y=row.yield_QED,mark='target_mean'))
        axes[0,j].set_title(NAMES[g],loc='left');axes[0,j].set_ylim(.88,1.025);axes[1,j].set_ylim(.4,.76)
        for ax in axes[:,j]:
            ax.set_xlim(.8,6.2);ax.set_xticks(range(1,7));ax.axvline(3.5,color='#94A3B8',ls='--',lw=.8)
            ax.grid(color='#E2E8F0',lw=.6);ax.set_axisbelow(True)
            ax.set_xlabel('Saved candidate positions inspected')
    axes[0,0].set_ylabel('Delivered / 125 planned runs');axes[1,0].set_ylabel('QED per planned run')
    handles=[Line2D([0],[0],color='#334155',ls=styles[p][0],marker=styles[p][1],label=l) for p,l in zip(POLICIES,LABELS)]
    fig.legend(handles=handles,ncol=3,loc='outside lower center',fontsize=9,frameon=False)
    save(fig,'03_candidate_count',[r/'policy_summary.csv'],
         'Chronological saved-candidate prefixes, no new generation. Up to three round1 positions followed by three round2 positions generated with full round1 feedback. Counts printed above delivery points. These are not API-cost or reduced-generation-budget experiments.')
    pd.DataFrame(points).to_csv(out/'plotted_coordinates.csv',index=False,encoding='utf-8-sig')
    (out/'figure_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Three scientific figures rendered, with source hashes and all plotted coordinates.')


if __name__=='__main__':main()
