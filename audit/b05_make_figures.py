"""Offline figures from frozen JSON; no network, training, or docking."""
import argparse
import json
from pathlib import Path

import os
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / 'matplotlib_cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.font_manager import FontProperties
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import Descriptors, QED, Draw

COLORS = {'T':'#687785','M':'#1976ad','G':'#8d55a8','NG':'#687785',
          'TG':'#bc7b19','PG-aggr':'#1976ad','PG-mild':'#8d55a8'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.titlesize':14,
                     'axes.labelsize':13,'pdf.fonttype':42,'svg.fonttype':'none',
                     'axes.spines.top':False,'axes.spines.right':False})


def save(fig, name, output):
    fig.canvas.draw()
    for ax in fig.axes:
        for title,body,bounds in getattr(ax,'_b05_boxes',[]):
            a=title.get_window_extent();b=body.get_window_extent()
            x,y,w,h=bounds
            left,bottom=ax.transData.transform((x,y));right,top=ax.transData.transform((x+w,y+h))
            assert min(a.x0,b.x0)>=left-1 and max(a.x1,b.x1)<=right+1,(name,'horizontal overflow')
            assert b.y0>=bottom-1 and a.y1<=top+1 and b.y1<a.y0,(name,'vertical overlap')
    for ext in ['pdf','svg','png']:
        fig.savefig(output / (name+'.'+ext),dpi=210,bbox_inches='tight',pad_inches=.16)
    plt.close(fig)


def box(ax, x,y,w,h, title, body, color='#e8f1f6', size=13):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.008',
                 facecolor=color,edgecolor='#8e9aa5',linewidth=1))
    renderer=ax.figure.canvas.get_renderer()
    maxpx=(ax.transData.transform((x+w,y))[0]-ax.transData.transform((x,y))[0])*.93
    def wrapped(text,fontsize,weight='normal'):
        font=FontProperties(family='DejaVu Sans',size=fontsize,weight=weight)
        lines=[]
        for paragraph in text.split('\n'):
            current=''
            for word in paragraph.split():
                candidate=(current+' '+word).strip()
                if current and renderer.get_text_width_height_descent(candidate,font,False)[0]>maxpx:
                    lines.append(current);current=word
                else:current=candidate
            lines.append(current)
        return '\n'.join(lines)
    title_text=ax.text(x+w/2,y+h-.025,wrapped(title,size,'bold'),ha='center',va='top',weight='bold',fontsize=size)
    body_text=ax.text(x+w/2,y+h*.40,wrapped(body,size-1),ha='center',va='center',fontsize=size-1,linespacing=1.30)
    if not hasattr(ax,'_b05_boxes'):ax._b05_boxes=[]
    ax._b05_boxes.append((title_text,body_text,(x,y,w,h)))


def arrow(ax, start, end, color='#53636f'):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='-|>',lw=1.35,color=color))


def flow(output):
    fig, ax = plt.subplots(figsize=(11.3,7.2)); fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    box(ax,.015,.77,.25,.20,'Task input','Target text + literature\nSeed SMILES\nRequested RC = seed RC − 1')
    box(ax,.305,.77,.34,.20,'Optional exemplars','T: none | M: Morgan | G: frozen GIN\nSame exact-RC pool first\nCosine ranking → 3 references')
    box(ax,.695,.77,.29,.20,'Shared five-turn dialogue','Turns 1–3: contextual reasoning\nTurn 4: 3 proposal positions\nTurn 5: 3 proposal positions')
    arrow(ax,(.265,.87),(.305,.87)); arrow(ax,(.645,.87),(.695,.87))
    box(ax,.365,.43,.32,.22,'Compute after turn 4','RDKit: parse, fragments, RC\nQED, SA and exact-copy flags\nSend computed turn-5 feedback',color='#e7f2eb')
    arrow(ax,(.84,.77),(.84,.63)); arrow(ax,(.84,.63),(.687,.54))
    ax.text(.86,.66,'turn 4',fontsize=12,ha='left')
    arrow(ax,(.53,.65),(.53,.72)); arrow(ax,(.53,.72),(.82,.72)); arrow(ax,(.82,.72),(.82,.77))
    ax.text(.67,.685,'feedback',fontsize=12,ha='center')
    box(ax,.015,.13,.28,.21,'Retain qualified union','Turns 4 + 5: up to 6 positions\nSingle fragment; exact target RC\nNo exact seed/reference copy')
    box(ax,.36,.13,.28,.21,'Deterministic delivery','Max QED in qualified union\nSA breaks equal-QED ties\nEmpty union → no delivery')
    box(ax,.705,.13,.28,.21,'Separate evaluation','Delivery S and QED yield U\nTarget-paired comparisons\nCalibrated docking afterward',color='#f5eee2')
    arrow(ax,(.984,.85),(.997,.85)); arrow(ax,(.997,.85),(.997,.385))
    arrow(ax,(.997,.385),(.155,.385)); arrow(ax,(.155,.385),(.155,.34))
    ax.text(.735,.40,'turns 4 + 5 candidates',fontsize=12,ha='center')
    arrow(ax,(.295,.235),(.36,.235)); arrow(ax,(.64,.235),(.705,.235))
    ax.text(.02,.04,'Invalid / missing tags consume positions. Bounded technical retries; no quality-driven resampling.\nVina is separate from generation/selection. Current Study B is distinct from historical Study A.',fontsize=12,va='bottom')
    save(fig,'fig1_current_pipeline',output)


def history(p,output):
    fig, axes=plt.subplots(1,2,figsize=(10.4,4.1),layout='constrained')
    groups=['NG','TG','PG-aggr','PG-mild']
    for j,g in enumerate(groups):
        rows=[r for r in p['A_targets'] if r['group']==g]
        vals=np.array([r['dRC_seed_minus_output'] for r in rows])
        offsets=np.linspace(-.12,.12,len(rows))
        axes[0].scatter(np.full(len(vals),j)+offsets,vals,s=20,alpha=.55,color=COLORS[g])
        axes[0].scatter(j,vals.mean(),s=90,marker='D',color=COLORS[g],edgecolors='white',zorder=4)
        axes[1].scatter([r['sim'] for r in rows],[r['QED'] for r in rows],s=24,alpha=.5,color=COLORS[g])
        axes[1].scatter(np.mean([r['sim'] for r in rows]),np.mean([r['QED'] for r in rows]),s=95,marker='D',color=COLORS[g],edgecolors='black',lw=.6,label=g)
    axes[0].axhline(1,color='#999999',ls='--',lw=1,label='one fewer ring')
    axes[0].set(xticks=range(4),xticklabels=groups,ylabel='Ring reduction: seed − output',title='(a) Historical strategy response')
    axes[0].legend(fontsize=11,loc='upper left')
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Mean Morgan Tanimoto to seed',ylabel='Mean QED',title='(b) Seed resemblance versus QED')
    axes[1].legend(fontsize=11,loc='lower right')
    fig.suptitle('Study A: 25 target means / group; 500 retained historical records',fontsize=12)
    save(fig,'fig2_historical_tradeoffs',output)


def mechanism(p,output):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.3),layout='constrained')
    feedback={r['chemblid']:r['yield_QED'] for r in p['C_targets'] if r['branch']=='Feedback'}
    control={r['chemblid']:r['yield_QED'] for r in p['C_targets'] if r['branch']=='NoExtraFeedback'}
    ids=sorted(feedback); assert len(ids)==25 and set(ids)==set(control)
    axes[0].plot([0,1],[0,1],color='#999999',ls='--',lw=1)
    axes[0].scatter([control[i] for i in ids],[feedback[i] for i in ids],color=COLORS['M'],s=30,alpha=.7)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='NoExtraFeedback target QED yield',ylabel='Feedback target QED yield',title='(a) New paired fifth-turn branches')
    r=p['facts']['C']['primary']
    axes[0].text(.03,.96,f"ΔU = {r['difference']:+.4f}\n95% CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]\np = {r['p_raw']:.4f}; n = 25 targets",transform=axes[0].transAxes,va='top',fontsize=11,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
    replay=p['facts']['replay']
    for g in ['T','M','G']:
        axes[1].plot(range(4),[r[g]['U'] for r in replay],marker='o',color=COLORS[g],label=g)
    axes[1].set(ylim=(0,1),xticks=range(4),xticklabels=['Turn 4\nQED','Turn 5\nQED','Union\nfirst eligible','Union\nQED'],ylabel='QED yield per planned position',title='(b) Fixed-candidate replay (descriptive)')
    axes[1].text(.02,.98,'Budget: 3 / 3 / 6 / 6 positions\nSaved attempts; not new ablation',transform=axes[1].transAxes,va='top',fontsize=11)
    axes[1].legend(ncol=3,loc='lower right',fontsize=11)
    save(fig,'fig3_mechanism',output)


def downstream(p,output):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.4),layout='constrained',gridspec_kw={'width_ratios':[1,1.1]})
    means={(r['chemblid'],r['group']):r['compliant_QED_utility'] for r in p['B_targets']}
    ids=sorted({i[0] for i in means}); delta=np.array([means[i,'G']-means[i,'M'] for i in ids]); assert len(delta)==25
    axes[0].scatter(range(1,26),delta,color=COLORS['G'],s=27)
    axes[0].axhline(0,color='#777777',lw=1); axes[0].axhline(delta.mean(),color=COLORS['G'],ls='--',lw=1)
    axes[0].set(xlabel='Target index (lexicographic ID order)',ylabel='GIN − Morgan target QED yield',title='(a) Primary G − M: 25 paired targets')
    stats=[r for r in p['facts']['B']['statistics'] if r['metric']=='U']
    for y,r in enumerate(stats):
        c=COLORS['G'] if r['comparison']=='G-M' else COLORS['M']
        axes[1].errorbar(r['difference'],y,xerr=[[r['difference']-r['ci_low']],[r['ci_high']-r['difference']]],fmt='o',color=c,capsize=4)
        axes[1].text(.155,y,f"pH = {r['p_holm']:.4f}",va='center',fontsize=12)
    axes[1].axvline(0,color='#777777',lw=1)
    axes[1].set(yticks=range(3),yticklabels=['G − M\nprimary','G − T\nauxiliary','M − T\nauxiliary'],ylim=(2.6,-.6),xlim=(-.07,.23),xlabel='Target-mean QED yield difference',title='(b) Contrasts: unadjusted 95% CI')
    axes[1].text(.03,.025,'pH: original family-specific Holm p\nNonsignificant ≠ equivalent',transform=axes[1].transAxes,fontsize=11)
    save(fig,'fig4_downstream',output)


def docking(p,output):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.5),layout='constrained')
    for ax,study,groups in [(axes[0],'A',['NG','TG','PG-aggr','PG-mild']),(axes[1],'B',['T','M','G'])]:
        for g in groups:
            rows=[r for r in p['dock_target'] if r['study']==study and r['group']==g]; assert len(rows)==11
            ax.scatter([r['dVina_output_minus_seed'] for r in rows],[r['dQED_active_output_minus_seed'] for r in rows],s=22,alpha=.40,color=COLORS[g])
            r=next(r for r in p['facts']['docking']['groups'] if r['study']==study and r['group']==g)
            ax.errorbar(r['delta_Vina'],r['delta_active_QED'],xerr=[[r['delta_Vina']-r['ci_low']],[r['ci_high']-r['delta_Vina']]],fmt='D',color=COLORS[g],capsize=3,ms=7,label=g)
        ax.axhline(0,color='#999999',lw=.8); ax.axvline(0,color='#999999',lw=.8)
        ax.set(xlabel='ΔVina: output − seed (kcal/mol)\nPositive is less favorable',ylabel='ΔQED: output − seed\n(organic component)',title=f'Study {study}: 11 calibrated targets')
        ax.legend(fontsize=11,loc='lower right')
    axes[0].set_xlim(-3.5,5); axes[1].set_xlim(-3.5,5)
    axes[0].set_ylim(-.4,.8); axes[1].set_ylim(-.4,.8)
    save(fig,'fig5_docking_tradeoffs',output)


def gin(output):
    fig,ax=plt.subplots(figsize=(11.2,6.6)); fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    box(ax,.015,.69,.285,.24,'Triplet-builder code','Morgan: radius 2 / 2048 bits\nL2 cosine; top-k (default 10)\nWithin neighbors: min / max RC\nPositive = min; negative = max',size=13)
    box(ax,.355,.69,.28,.24,'Training implementation','No generic pretrained load\n15,461 retained RC triplets\n12,368 train / 3,093 validation\nOriginal launch chain incomplete',size=13)
    box(ax,.69,.69,.295,.24,'Batch contrastive loss','Anchor i targets positive i\nAll b positives + b negatives\nUnnormalized dot products\nFull 2b denominator at τ',size=13)
    arrow(ax,(.30,.81),(.355,.81)); arrow(ax,(.635,.81),(.69,.81))
    box(ax,.015,.29,.27,.25,'Molecular graph input','84 atom features\nBidirectional bond adjacency\n9 edge attributes saved\nEdge attributes NOT consumed',color='#f5eee2')
    box(ax,.365,.29,.27,.25,'Frozen GIN encoder','5 × GINConv / BN / ReLU\nWidth = 256; learnable ε\nMean pooling → 256-vector\nEvaluation mode',color='#e7f2eb')
    box(ax,.715,.29,.27,.25,'Downstream inference','L2 normalization → cosine\nExact-RC pool selected first\nTop 3 references per target\nFrozen during generation',color='#e7f2eb')
    arrow(ax,(.285,.42),(.365,.42)); arrow(ax,(.635,.42),(.715,.42))
    ax.text(.5,.13,r'$\mathcal{L}=-\frac{1}{b}\sum_i\log\frac{\exp(a_i^\top p_i/\tau)}{\sum_j[\exp(a_i^\top p_j/\tau)+\exp(a_i^\top n_j/\tau)]}$',ha='center',fontsize=17)
    ax.text(.02,.025,'Defaults are not the exact original run. Positive need not improve on the anchor.\nDeployed GIN is not the later P0/P1 chain; no new architecture or pretraining benefit is established.',fontsize=12)
    save(fig,'figS1_gin_implementation',output)


def reference(p,output):
    fig,axes=plt.subplots(1,2,figsize=(10.4,4),layout='constrained')
    data=p['reference_rows']; assert len(data)==150
    counts={g:{rc:sum(r['group']==g and r['RC']==rc for r in data) for rc in range(1,7)} for g in ['M','G']}
    assert counts['M']==counts['G']
    for g,offset in [('M',-.18),('G',.18)]:
        vals=[r['tanimoto'] for r in data if r['group']==g]; assert len(vals)==75
        axes[0].bar(np.arange(1,7)+offset,list(counts[g].values()),width=.34,color=COLORS[g],label=g)
        sorted_vals=np.sort(vals); axes[1].step(sorted_vals,np.arange(1,76)/75,where='post',color=COLORS[g],label=f"{g}: mean {np.mean(vals):.4f}")
    axes[0].set(xticks=range(1,7),xlabel='Reference RDKit ring count',ylabel='Reference records',title='(a) Same RC: 75 records per backend')
    axes[0].legend(fontsize=12)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Common Morgan Tanimoto to seed',ylabel='Empirical cumulative fraction',title='(b) Tanimoto is not matched')
    axes[1].legend(loc='lower right',fontsize=12)
    save(fig,'figS2_reference_controls',output)


def chemistry(p,output):
    molecules=[];legends=[]; rows=[]
    for item in p['case_molecules']:
        mol=Chem.MolFromSmiles(item['smiles']); assert mol is not None
        rc=int(Descriptors.RingCount(mol)); qed=QED.qed(mol)
        if 'RC' in item: assert rc==item['RC'] and abs(qed-item['QED'])<1e-9
        rows.append(dict(**item,recomputed_RC=rc,recomputed_QED=qed))
        molecules.append(mol); legends.append(f"{item['label']}  |  RC = {rc}; QED = {qed:.4f}")
    options=Draw.MolDrawOptions();options.legendFontSize=24;options.legendFraction=.18
    options.fixedBondLength=27
    svg=Draw.MolsToGridImage(molecules,molsPerRow=2,subImgSize=(620,310),legends=legends,useSVG=True,drawOptions=options)
    (output/'figS3_structural_example.svg').write_text(svg,encoding='utf-8')
    png=Draw.MolsToGridImage(molecules,molsPerRow=2,subImgSize=(620,310),legends=legends,drawOptions=options)
    png.save(output/'figS3_structural_example.png')
    fig,ax=plt.subplots(figsize=(8.2,10.3));ax.imshow(png);ax.axis('off');fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.savefig(output/'figS3_structural_example.pdf',dpi=300,bbox_inches='tight',pad_inches=0);plt.close(fig)
    return rows


def toc(output):
    fig,ax=plt.subplots(figsize=(10.3,3));fig.subplots_adjust(left=.01,right=.99,bottom=.02,top=.99);ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    entries=[('Condition','Target text + seed\nExactly one fewer ring'),('Optional examples','Same eligible pool\nMorgan or frozen GIN'),('Compute + retain','Program feedback\nQualified union'),('Deliver + evaluate','QED ranking / failure\nDocking trade-off')]
    for i,(title,body) in enumerate(entries):
        x=.015+i*.25;box(ax,x,.24,.22,.57,title,body,size=13)
        if i<3:arrow(ax,(x+.22,.52),(x+.25,.52))
    ax.text(.5,.065,'Conditioned generation; neither GIN superiority nor validated affinity is established',ha='center',fontsize=12)
    save(fig,'graphical_overview',output)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    output=Path(args.output).resolve()
    local_output = Path(__file__).resolve().parents[1] / 'figures'
    assert str(output).startswith('/tmp/neurocomputing-b05-20261013-') or output == local_output, 'Use the dedicated B05 figure directory'
    output.mkdir(parents=True,exist_ok=True);p=json.loads(Path(args.input).read_text(encoding='utf-8'))
    flow(output);history(p,output);mechanism(p,output);downstream(p,output);docking(p,output)
    gin(output);reference(p,output);case=chemistry(p,output);toc(output)
    metadata=dict(main_figures=5,supplementary_figures=3,graphical_overviews=1,
                  diagram_box_bounds_checked=17,
                  plotting_versions=dict(matplotlib=matplotlib.__version__,numpy=np.__version__,rdkit=rdBase.rdkitVersion),
                  target_unit=25,docking_target_unit=11,reference_records=150,
                  structure_example_target=p['case_target'],case_rule=p['case_rule'],case=case,
                  chemical_structure_PDF='high-resolution RDKit depiction; separate native SVG retained',
                  new_API_calls=0,new_training_jobs=0,new_docking_jobs=0)
    (output/'figure_validation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('5 main figures, 3 SI figures and graphical overview generated offline; all ten structures validated.')


if __name__=='__main__':main()
