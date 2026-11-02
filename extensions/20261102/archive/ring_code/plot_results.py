from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
A=ROOT/'analysis';OUT=A/'figures';OUT.mkdir(exist_ok=True)
target=pd.read_csv(A/'target_means100.csv');summary=pd.read_csv(A/'group_summary4.csv');refs=pd.read_csv(ROOT/'inputs/references225.csv')
colors={'T':'#707b84','M':'#2874a6','G':'#8056a3','R':'#cc7b2b'};order=['T','M','G','R'];coords=[];rng=np.random.default_rng(42)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,2,figsize=(9.2,3.65),constrained_layout=True)
for ax,metric,title in zip(axes,['S','U'],['Qualified delivery rate','QED yield per planned run']):
    for j,g in enumerate(order):
        f=target[target.group==g].sort_values('chemblid');x=j+rng.uniform(-.14,.14,len(f));v=f[metric].to_numpy()
        ax.scatter(x,v,s=20,alpha=.48,color=colors[g],edgecolors='none')
        boots=np.random.default_rng(42).integers(0,len(v),(10000,len(v)));lo,hi=np.quantile(v[boots].mean(axis=1),[.025,.975]);mean=v.mean()
        ax.errorbar([j],[mean],yerr=[[mean-lo],[hi-mean]],fmt='D',color=colors[g],ecolor='#263238',markersize=6,capsize=4,zorder=5)
        for xx,(_,r) in zip(x,f.iterrows()):coords.append(dict(figure='01_outcomes',panel=metric,kind='target',group=g,chemblid=r.chemblid,x=xx,y=r[metric]))
        for kind,value in [('mean',mean),('CI_low',lo),('CI_high',hi)]:coords.append(dict(figure='01_outcomes',panel=metric,kind=kind,group=g,x=j,y=value))
    ax.set_xticks(range(4),['T: no examples','M: Morgan','G: GIN','R: random']);ax.tick_params(axis='x',labelsize=8.5)
    ax.set_title(title,pad=10);ax.set_ylim(-.025,1.045);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(OUT/'01_outcomes.png',dpi=240);fig.savefig(OUT/'01_outcomes.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(9.2,3.65),constrained_layout=True)
for j,g in enumerate(['M','G','R']):
    f=refs[refs.group==g];counts=f.RC.value_counts()
    for rc in range(1,int(refs.RC.max())+1):
        n=int(counts.get(rc,0));x=rc+(j-1)*.23;axes[0].bar(x,n,width=.21,color=colors[g],label=g if rc==1 else None)
        coords.append(dict(figure='02_reference_control',panel='ring_count',kind='count',group=g,x=x,y=n,RC=rc))
    x=j+rng.uniform(-.14,.14,len(f));axes[1].scatter(x,f.tanimoto,s=16,alpha=.5,color=colors[g],edgecolors='none')
    axes[1].boxplot([f.tanimoto.to_numpy()],positions=[j],widths=.4,showfliers=False,
        medianprops={'color':'black'},boxprops={'color':colors[g]},whiskerprops={'color':colors[g]},capprops={'color':colors[g]})
    for xx,(_,r) in zip(x,f.iterrows()):coords.append(dict(figure='02_reference_control',panel='tanimoto',kind='reference',group=g,chemblid=r.chemblid,rank=r['rank'],x=xx,y=r.tanimoto))
axes[0].set(title='Reference ring counts',xlabel='RDKit ring count',ylabel='Number of reference molecules');axes[0].legend(frameon=False)
axes[0].set_xticks(range(1,int(refs.RC.max())+1));axes[1].set(title='Reference similarity to the seed',ylabel='Morgan Tanimoto')
axes[1].set_xticks(range(3),['M: Morgan','G: GIN','R: random']);axes[1].set_ylim(0,1)
for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
fig.savefig(OUT/'02_reference_control.png',dpi=240);fig.savefig(OUT/'02_reference_control.svg');plt.close(fig)
pd.DataFrame(coords).to_csv(OUT/'plotted_coordinates.csv',index=False)
sources=[A/'target_means100.csv',A/'group_summary4.csv',ROOT/'inputs/references225.csv',OUT/'plotted_coordinates.csv']
index=dict(source_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
    coordinate_rows=len(coords),mean_CIs='Unadjusted target bootstrap intervals; infer group contrasts from declared_tests8.csv.',
    figures={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.suffix in ['.png','.svg']})
(OUT/'figure_index.json').write_text(json.dumps(index,indent=2)+'\n')
tests=pd.read_csv(A/'declared_tests8.csv');refsummary=pd.read_csv(A/'reference_distribution_summary.csv')
interpret=['','## 这次对照说明什么','',
    '主要比较直接回答：同样给三个达到环数要求的范例，按Morgan/GIN选择是否比本轮固定随机范例更有帮助。R是随机输入范例，不是随机挑最终输出。','']
for comparison,label in [('M-R','Morgan相对随机范例'),('G-R','GIN相对随机范例'),('R-T','随机范例相对无范例'),('G-M','GIN相对Morgan')]:
    r=tests[(tests.comparison==comparison)&(tests.metric=='U')].iloc[0]
    if r.p_holm<.05 and r.difference>0:result='本轮支持前者更高的QED产出'
    elif r.p_holm<.05 and r.difference<0:result='本轮前者的QED产出更低'
    else:result='本轮未确认两者的QED产出差异'
    interpret.append(f"- {label}：ΔU={r.difference:+.4f}，校正p={r.p_holm:.5f}；{result}。")
interpret+=['','这些判断分别对应事前规定的主要或次要检验族。未确认差异不能写成证明等效，结果的适用范围是本轮减环条件和这25个任务。',
    '','| 参考组 | 每任务参考数 | 平均环数 | 平均种子Tanimoto | 平均QED |','|---|---:|---:|---:|---:|']
for group in ['M','G','R']:
    r=refsummary[refsummary.group==group].iloc[0]
    interpret.append(f"| {group} | 3 | {r.RC_mean:.2f} | {r.tanimoto_mean:.4f} | {r.QED_mean:.4f} |")
interpret+=['','三个参考组在每个任务的环数条件完全一致，来自相同候选池。Tanimoto和QED并未人为配平，其分布随所定义的选例方式变化。原始逐分子值与分布统计均已保存。',
    '','只看最高QED最终输出可能看不到候选与选择的区别；candidate_quality4.csv保留候选层面性质，policy_summary72.csv同时提供首个合格、均匀随机合格期望和最高QED的重放。','',
    '本次补充完成了原减环任务所缺的随机范例对照。原375位置独立保留；不能将本轮R行直接拼入旧375位置的表格，并沿用旧批次p值或Vina分数。']
with (A/'结果与解读.md').open('a',encoding='utf-8') as f:f.write('\n'.join(interpret)+'\n')
print(json.dumps(dict(status='plots_ready',coordinate_rows=len(coords),figures=4)))
