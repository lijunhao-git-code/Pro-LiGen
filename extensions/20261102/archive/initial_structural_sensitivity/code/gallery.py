"""Create a fully offline, per-target molecular inspection page with real matched atoms."""
import hashlib
import html
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT/'gallery'
    (target/'svg').mkdir(parents=True, exist_ok=True)
    records = pd.read_csv(ROOT/'results/final_outputs_and_baselines450.csv').where(lambda x:x.notna(), None).to_dict('records')
    cache = {r['key']:r for r in map(json.loads,(ROOT/'results/mcs_cache.jsonl').read_text(encoding='utf-8').splitlines())}
    drawn, payload = {}, []
    for row in records:
        image = None
        if row['delivery']:
            key = row['mcs_key']
            mcs = cache[key]
            if key not in drawn:
                mols = [Chem.MolFromSmiles(mcs['seed_organic']), Chem.MolFromSmiles(mcs['candidate_organic'])]
                query = Chem.MolFromSmarts(mcs['smarts'])
                atoms = [list(mol.GetSubstructMatch(query)) for mol in mols]
                assert all(len(v)==mcs['common_atoms'] for v in atoms)
                bonds=[]
                for mol, match in zip(mols,atoms):
                    bonds.append([mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()],match[b.GetEndAtomIdx()]).GetIdx() for b in query.GetBonds()])
                drawer = rdMolDraw2D.MolDraw2DSVG(800,310,400,310)
                opts=drawer.drawOptions()
                opts.legendFontSize=17
                opts.padding=.13
                drawer.DrawMolecules(mols,legends=['Seed organic component','Selected organic component'],
                                     highlightAtoms=atoms,highlightBonds=bonds)
                drawer.FinishDrawing()
                (target/'svg'/(key+'.svg')).write_text(drawer.GetDrawingText(),encoding='utf-8')
                drawn[key]=len(atoms[0])
            image='svg/'+key+'.svg'
        payload.append({k:row.get(k) for k in ['method','chemblid','slot_id','position','delivery','QED','SA_raw',
                       'tanimoto_parent','mcs_seed_fraction','mcs_lower_seed_fraction','mcs_complete',
                       'dHA_organic','dMW_organic','dQED_parent','canonical','seed_organic_canonical']})
        payload[-1]['image']=image
    # pandas float columns can retain NaN after where(None); serialize recursively as genuine null.
    def clean(v):
        if isinstance(v,dict): return {k:clean(x) for k,x in v.items()}
        if isinstance(v,list): return [clean(x) for x in v]
        if isinstance(v,float) and pd.isna(v): return None
        return v
    data=json.dumps(clean(payload),ensure_ascii=False,allow_nan=False).replace('</','<\/')
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>分子结构人工核查</title><style>body{font-family:system-ui,sans-serif;background:#f3f5f7;margin:24px;color:#172330;max-width:1200px}h1{font-size:24px}p{line-height:1.7}select{padding:8px;margin-right:16px;font-size:15px}article{background:white;padding:18px;margin:18px 0;border:1px solid #d5dde4;border-radius:9px}img{width:100%;max-width:850px;display:block}h2{font-size:18px;margin:0 0 10px}.stats{font-variant-numeric:tabular-nums;line-height:1.8}.smiles{word-break:break-all;color:#435267;font-family:monospace;font-size:12px}.note{color:#506273}summary{cursor:pointer}</style>
<h1>分子结构人工核查</h1><p>选择一个任务，逐个查看生成结果和直接检索结果。每行图左边是种子有机组分，右边是最终分子有机组分；着色部分是搜索得到的共同子结构。所有图片来自下方记录的真实 SMILES，无示意分子。只有 CHEMBL1821 种子含一个单原子溴对离子，结构图移除该对离子；QED 与原论文的 Tanimoto 仍用原始分子形式。</p>
<p class="note">共同原子覆盖率 = MCS 原子数 ÷ 种子有机组分重原子数。它允许环与链对应，忽略手性与形式电荷是否一致，用于描述保留程度。未完成的 MCS 搜索显示“下界”。返回数为零的位置保留在记录中。</p>
<label>任务 <select id="task"></select></label><label>方法 <select id="method"><option value="all">全部方法</option></select></label><div id="count"></div><main id="cards"></main>
<script>const rows=__DATA__;const names={T:'T：无范例',M:'M：Morgan 范例',G:'G：GIN 范例',R_NN1:'直接检索：最近邻',R_QED3:'直接检索：前3个按QED选优',R_QED6:'直接检索：前6个按QED选优'};const task=document.querySelector('#task'),method=document.querySelector('#method');
for(const value of [...new Set(rows.map(r=>r.chemblid))].sort()){task.add(new Option(value,value));}for(const [k,v]of Object.entries(names)){method.add(new Option(v,k));}
const fmt=(v,n=3)=>v===null?'—':Number(v).toFixed(n);function text(el,s){el.textContent=s;return el;}function draw(){const chosen=rows.filter(r=>r.chemblid===task.value&&(method.value==='all'||r.method===method.value));document.querySelector('#count').textContent=`当前显示 ${chosen.length} 个计划位置`;const cards=document.querySelector('#cards');cards.replaceChildren();for(const r of chosen){const a=document.createElement('article');a.append(text(document.createElement('h2'),`${names[r.method]} · ${r.slot_id}`));if(r.image){const im=new Image();im.src=r.image;im.alt='种子与最终分子的共同子结构';im.loading='lazy';a.append(im);}const s=document.createElement('p');s.className='stats';s.textContent=r.delivery?`QED ${fmt(r.QED)}，相对种子 ${fmt(r.dQED_parent)}；SA ${fmt(r.SA_raw)}；Tanimoto ${fmt(r.tanimoto_parent)}；共同原子覆盖率 ${fmt(r.mcs_complete?r.mcs_seed_fraction:r.mcs_lower_seed_fraction)}${r.mcs_complete?'':'（搜索未完成，下界）'}；有机重原子变化 ${fmt(r.dHA_organic,0)}；有机分子量变化 ${fmt(r.dMW_organic,1)} Da`:'此位置未交回合格分子；仍计入计划分母。';a.append(s);if(r.canonical){const d=document.createElement('details');d.append(text(document.createElement('summary'),'查看真实 SMILES'));const p=document.createElement('p');p.className='smiles';p.textContent='最终分子：'+r.canonical;d.append(p);const q=document.createElement('p');q.className='smiles';q.textContent='种子有机组分：'+r.seed_organic_canonical;d.append(q);a.append(d);}cards.append(a);}}
task.onchange=method.onchange=draw;draw();</script></html>'''
    (target/'index.html').write_text(page.replace('__DATA__',data),encoding='utf-8')
    (target/'gallery_verification.json').write_text(json.dumps(dict(displayed_positions=len(payload),drawn_unique_pairs=len(drawn),
        all_highlight_matches_equal_reported_MCS_atoms=True,source_sha256=hashlib.sha256((ROOT/'results/final_outputs_and_baselines450.csv').read_bytes()).hexdigest()),indent=2)+'\n',encoding='utf-8')
    print(f'Offline gallery: {len(payload)} positions; {len(drawn)} verified real molecular pairs.')


if __name__=='__main__':
    main()
