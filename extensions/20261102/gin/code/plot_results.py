"""Scientific figures drawn only from exported numeric tables, with a point/source index."""
import os,json
os.environ.setdefault('MPLCONFIGDIR','/tmp/proligen_property_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np,pandas as pd
from common import ROOT,sha,now
from analysis_utils import csv,save_json
from retained_statistics import interval
GROUPS=['T','M','G','R'];COLORS={'T':'#64748B','M':'#007F86','G':'#D77721','R':'#8862AA'}
NAMES={'T':'No examples','M':'Morgan','G':'GIN','R':'Random examples'}
PNAMES={'logp':'LogP: decrease by 1.0 ± 0.2','tpsa':'TPSA: decrease by 20 ± 5 Å²'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'axes.titlesize':11,'axes.labelsize':10,'svg.fonttype':'none'})


def main(base=None):
    base=base or ROOT/'analysis';out=base/'figures';out.mkdir(exist_ok=True);points=[];index=[]
    p=base/'properties';s=base/'structures';d=base/'docking'
    target=pd.read_csv(p/'target_means.csv');summary=pd.read_csv(p/'group_summary.csv')
    def save(fig,name,sources,note):
        for ext in ['png','svg']:fig.savefig(out/f'{name}.{ext}',dpi=240,bbox_inches='tight',pad_inches=.12)
        plt.close(fig);index.append(dict(figure=name,sources={str(file.relative_to(ROOT)):sha(file) for file in sources},note=note,
            output_sha256={ext:sha(out/f'{name}.{ext}') for ext in ['png','svg']}))
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,prop in enumerate(['logp','tpsa']):
        for j,metric,label in [(0,'S','Eligible delivery (%)'),(1,'U','Compliant QED yield')]:
            ax=axes[i,j]
            for x,group in enumerate(GROUPS):
                vals=target[target.property.eq(prop)&target.group.eq(group)][metric].to_numpy();assert len(vals)==25
                est=interval(vals);scale=100 if metric=='S' else 1
                mean,low,high=[est[k]*scale for k in ['difference','ci_low','ci_high']]
                ax.bar(x,mean,color=COLORS[group],width=.65,alpha=.9)
                ax.errorbar(x,mean,yerr=[[max(0,mean-low)],[max(0,high-mean)]],color='#222222',capsize=4,lw=1)
                row=summary[summary.property.eq(prop)&summary.group.eq(group)].iloc[0]
                annotation=f'{int(row.delivered)}/125' if metric=='S' else f'{mean:.3f}'
                ax.annotate(annotation,(x,high),xytext=(0,5),textcoords='offset points',ha='center',fontsize=9)
                points.append(dict(figure='01_primary_outcomes',property=prop,panel=metric,group=group,x=x,y=mean,ci_low=low,ci_high=high))
            ax.set_xticks(range(4),['T','M','G','R']);ax.set_ylabel(label);ax.set_title(PNAMES[prop],loc='left')
            ax.set_ylim(0,112 if metric=='S' else max(.8,float(summary.U.max())+.1));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend([Line2D([0],[0],color=COLORS[g],lw=5) for g in GROUPS],[f'{g}: {NAMES[g]}' for g in GROUPS],loc='outside lower center',ncol=4,frameon=False,fontsize=9)
    save(fig,'01_primary_outcomes',[p/'target_means.csv',p/'group_summary.csv'],'25 paired target means; five positions per target. Error bars are descriptive95% target bootstrap intervals. U includes failed positions as zero.')
    # Separating the proposal pool and final selection avoids calling best-of-six a proposal effect.
    policy=pd.read_csv(p/'policy_summary.csv');cq=pd.read_csv(p/'candidate_target_quality.csv')
    fig,axes=plt.subplots(2,3,figsize=(12,7.1),layout='constrained')
    for i,prop in enumerate(['logp','tpsa']):
        for j,field in [(0,'valid_QED'),(1,'eligible_fraction')]:
            ax=axes[i,j]
            for x,group in enumerate(GROUPS):
                part=cq[cq.property.eq(prop)&cq.group.eq(group)].sort_values('chemblid');values=part[field].to_numpy()
                xs=x+np.linspace(-.14,.14,len(values));scale=100 if field=='eligible_fraction' else 1
                ax.scatter(xs,values*scale,s=15,color=COLORS[group],alpha=.5,linewidths=0)
                ax.scatter(x,np.nanmean(values)*scale,s=48,color=COLORS[group],marker='D',edgecolor='black',linewidth=.5)
                for xx,(_,r) in zip(xs,part.iterrows()):points.append(dict(figure='02_pool_and_selection',property=prop,panel=field,group=group,chemblid=r.chemblid,x=xx,y=r[field]*scale))
            ax.set_xticks(range(4),GROUPS);ax.set_ylim(0,1 if field=='valid_QED' else 105)
            ax.set_ylabel('All-valid proposal QED' if field=='valid_QED' else 'Eligible proposal positions (%)')
            ax.set_title(PNAMES[prop],loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax=axes[i,2]
        for group in GROUPS:
            for pi,rule in enumerate(['first_eligible','random_unique_expected','best_QED']):
                r=policy[policy.property.eq(prop)&policy.group.eq(group)&policy.k.eq(6)&policy.policy.eq(rule)].iloc[0]
                points.append(dict(figure='02_pool_and_selection',property=prop,panel='selection_U',group=group,policy=rule,x=pi,y=r.U))
            sub=policy[policy.property.eq(prop)&policy.group.eq(group)&policy.k.eq(6)].set_index('policy')
            ax.plot(range(3),sub.loc[['first_eligible','random_unique_expected','best_QED'],'U'],marker='o',color=COLORS[group],label=group,lw=1.5)
        ax.set_xticks(range(3),['First','Random\nexpectation','Highest QED']);ax.set_ylabel('Compliant QED yield');ax.set_ylim(0,.9);ax.grid(alpha=.2);ax.legend(frameon=False,ncol=4,fontsize=8)
    save(fig,'02_pool_and_selection',[p/'policy_summary.csv',p/'candidate_target_quality.csv'],'Left/middle:25 target-level proposal statistics; right: same saved eligible pools under three selection rules. Random refers to output selection, distinct from the R input-example arm.')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,prop in enumerate(['logp','tpsa']):
        for j,metric,label in [(0,'S','Eligible delivery'),(1,'U','Compliant QED yield')]:
            ax=axes[i,j]
            for group in GROUPS:
                sub=policy[policy.property.eq(prop)&policy.group.eq(group)&policy.policy.eq('best_QED')].sort_values('k')
                assert len(sub)==6
                ax.plot(sub.k,sub[metric],color=COLORS[group],marker='o',lw=1.5,label=group)
                for _,r in sub.iterrows():points.append(dict(figure='03_saved_prefixes',property=prop,panel=metric,group=group,x=int(r.k),y=r[metric]))
            ax.set_xticks(range(1,7));ax.set_xlabel('First k saved proposal positions');ax.set_ylabel(label);ax.set_title(PNAMES[prop],loc='left');ax.set_ylim(0,1.05 if metric=='S' else .9);ax.grid(alpha=.2)
    axes[0,0].legend(frameon=False,ncol=4)
    save(fig,'03_saved_prefixes',[p/'policy_summary.csv'],'Chronological saved-candidate sensitivity, not independent generation under smaller API budgets. Positions1–3 precede program feedback;4–6 follow it.')
    refs=pd.read_csv(p/'reference_target_means.csv');fig,axes=plt.subplots(2,3,figsize=(12,6.8),layout='constrained')
    for i,prop in enumerate(['logp','tpsa']):
        for j,field,label in [(0,'target_error','Reference error from target center'),(1,'tanimoto','Reference Tanimoto to seed'),(2,'RC','Reference ring count')]:
            ax=axes[i,j]
            for x,group in enumerate(['M','G','R']):
                sub=refs[refs.property.eq(prop)&refs.group.eq(group)].sort_values('chemblid');xs=x+np.linspace(-.15,.15,25)
                ax.scatter(xs,sub[field],s=16,color=COLORS[group],alpha=.6,linewidths=0);ax.scatter(x,sub[field].mean(),marker='D',color=COLORS[group],s=45,edgecolor='black',lw=.5)
                for xx,(_,r) in zip(xs,sub.iterrows()):points.append(dict(figure='04_reference_control',property=prop,panel=field,group=group,chemblid=r.chemblid,x=xx,y=r[field]))
            if field=='target_error':
                tol=.2 if prop=='logp' else 5
                ax.axhline(-tol,color='#444444',ls='--',lw=.8);ax.axhline(tol,color='#444444',ls='--',lw=.8);ax.set_ylim(-tol*1.2,tol*1.2)
            if field=='tanimoto':ax.set_ylim(0,1)
            ax.set_xticks(range(3),['M','G','R']);ax.set_ylabel(label);ax.set_title(PNAMES[prop],loc='left');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'04_reference_control',[p/'reference_target_means.csv'],'Each dot is one target average over references and five runs. All arms use the same attribute-band pool; structural distributions need not be equal because selection differs. Full2,250 reference records are provided separately.')
    if (s/'structure_sensitivity_summary.csv').exists():
        curves=pd.read_csv(s/'structure_sensitivity_summary.csv');fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
        for i,prop in enumerate(['logp','tpsa']):
            for j,metric,label in [(0,'S','Delivery after structure filter'),(1,'U','Compliant QED yield')]:
                ax=axes[i,j]
                for group in GROUPS:
                    sub=curves[curves.property.eq(prop)&curves.group.eq(group)&curves.metric.eq('tanimoto')&curves.policy.eq('reselect_saved_eligible')].sort_values('threshold')
                    ax.plot(sub.threshold,sub[metric],color=COLORS[group],marker='o',lw=1.5,label=group)
                    for _,r in sub.iterrows():points.append(dict(figure='05_structure_sensitivity',property=prop,panel=metric,group=group,x=r.threshold,y=r[metric]))
                ax.set_xlabel('Minimum seed Tanimoto for saved-pool reselection');ax.set_ylabel(label);ax.set_ylim(0,1.05 if metric=='S' else .9);ax.set_title(PNAMES[prop],loc='left');ax.grid(alpha=.2)
        axes[0,0].legend(frameon=False,ncol=4)
        save(fig,'05_structure_sensitivity',[s/'structure_sensitivity_summary.csv'],'Descriptive Tanimoto thresholds on saved eligible candidates; no new generation. MCS completion-aware lower/upper curves and original-final filtering are in the numeric tables.')
    if (d/'docking_target_means.csv').exists():
        dt=pd.read_csv(d/'docking_target_means.csv');fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
        for ax,prop in zip(axes,['logp','tpsa']):
            for x,group in enumerate(GROUPS):
                sub=dt[dt.property.eq(prop)&dt.group.eq(group)&dt.kind.eq('generated')].sort_values('chemblid');xs=x+np.linspace(-.15,.15,len(sub))
                ax.scatter(xs,sub.delta_vina,s=22,color=COLORS[group],alpha=.65,linewidths=0);ax.scatter(x,sub.delta_vina.mean(),color=COLORS[group],marker='D',s=55,edgecolor='black',lw=.5)
                for xx,(_,r) in zip(xs,sub.iterrows()):points.append(dict(figure='06_docking_tradeoff',property=prop,panel='delta_vina',group=group,chemblid=r.chemblid,x=xx,y=r.delta_vina))
            ax.axhline(0,color='#444444',ls='--',lw=.9);ax.set_xticks(range(4),GROUPS);ax.set_ylabel('ΔVina from seed (kcal/mol)');ax.set_title(PNAMES[prop],loc='left');ax.grid(axis='y',alpha=.2)
        save(fig,'06_docking_tradeoff',[d/'docking_target_means.csv'],'Each dot is the available-output mean for one of11 prevalidated targets. Positive output-minus-seed is less favorable. Docking and delivery completion counts accompany these conditional scores in the tables.')
    csv(out/'plotted_coordinates.csv',points)
    save_json(out/'figure_index.json',dict(time_utc=now(),figures=index,coordinates_sha256=sha(out/'plotted_coordinates.csv'),
        all_values_come_from_exported_numeric_tables=True,new_API_calls=0))
    print('FIGURES',len(index),'COORDINATES',len(points),flush=True)


if __name__=='__main__':main()
