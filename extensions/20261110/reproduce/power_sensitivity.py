"""Recompute paired-t MDE sensitivity benchmarks from the frozen Study-B task means.

Requires Python >=3.10 and NumPy. No generation, training, docking or network calls.
Default: verify committed results. --write: generate the numerical records.
This is a paired-t model sensitivity analysis, not power for the sign-flip/Holm tests.
"""
from pathlib import Path
import argparse, csv, hashlib, json, math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/updated20261102'
DEST = ROOT / 'data/power20261109'

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def root_bisect(fn, target, low, high, increasing=True):
    for _ in range(65):
        mid=(low+high)/2
        if (fn(mid)<target)==increasing: low=mid
        else: high=mid
    return (low+high)/2

def calibration(order):
    # T=(Z+lambda)/sqrt(V/24), Z~N(0,1), V~chi-square(24).
    # Integrate over V in [0,200]; the omitted upper tail is negligible.
    z,w=np.polynomial.legendre.leggauss(order)
    v=(z+1)*100
    weights=w*100*np.exp(11*np.log(v)-v/2-12*np.log(2)-math.lgamma(12))
    q=np.sqrt(v/24)
    assert abs(float(sum(weights))-1)<1e-10
    def power(critical,nc):
        tails=np.array([.5*math.erfc((critical*x-nc)/math.sqrt(2))+
                        .5*math.erfc((critical*x+nc)/math.sqrt(2)) for x in q])
        return float(weights@tails)
    results={}
    for size in [1,4,8]:
        alpha=.05/size
        critical=root_bisect(lambda x:power(x,0),alpha,0,10,False)
        dz=root_bisect(lambda x:power(critical,5*x),.8,0,3)
        assert abs(power(critical,5*dz)-.8)<1e-10
        results[size]=dict(alpha=alpha,critical_t=critical,standardized_MDE=dz)
    assert abs(results[1]['critical_t']-2.063898561628)<1e-10
    return results

def calculate():
    cal=calibration(256);check=calibration(128)
    for m in cal:
        for key in cal[m]: assert abs(cal[m][key]-check[m][key])<1e-8
    task={}
    inputs=[SOURCE/'ring/target_means.csv',SOURCE/'properties/target_means.csv',SOURCE/'figure5_comparisons9.csv']
    for path in inputs[:2]:
        for r in read(path): task[(r.get('property','ring'),r['chemblid'],r['group'])]=float(r['U'])
    original={(r['condition'],r['comparison']):r for r in read(inputs[2])}
    summaries=[];differences=[]
    for condition in ['ring','logp','tpsa']:
        family_size=4 if condition=='ring' else 8
        for contrast in ['M-R','G-R','G-M']:
            a,b=contrast.split('-')
            ids=sorted(k[1] for k in task if k[0]==condition and k[2]==a)
            assert len(ids)==25
            d=np.array([task[condition,i,a]-task[condition,i,b] for i in ids])
            sd=float(d.std(ddof=1));mean=float(d.mean())
            assert abs(mean-float(original[condition,contrast]['difference']))<1e-12
            summaries.append(dict(condition=condition,contrast=contrast,metric='U',tasks=25,
                mean_difference=mean,sd_task_difference=sd,SE_task_difference=sd/5,
                unadjusted_alpha=.05,MDE80_unadjusted=cal[1]['standardized_MDE']*sd,
                family_size=family_size,conservative_alpha=.05/family_size,
                MDE80_conservative=cal[family_size]['standardized_MDE']*sd))
            for i,value in zip(ids,d): differences.append(dict(condition=condition,contrast=contrast,chemblid=i,U_difference=float(value)))
    meta=dict(method='Two-sided paired-t model sensitivity using observed SD of 25 paired task-mean U differences',
        interpretation='Conditional variance-based MDE benchmarks, not observed-effect power or exact power of the sign-flip/Holm analysis',
        task_count=25,degrees_of_freedom=24,target_power=.8,
        multiplicity='alpha=0.05/m with m=4 for ring and m=8 for joint LogP/TPSA families; a conservative Bonferroni reference, not a reproduction of Holm power',
        integration='Normal/chi-square mixture defining noncentral t; Gauss-Legendre quadrature [0,200]; orders 128 and 256 agree within 1e-8',
        calibrations=cal,software=dict(python='>=3.10',numpy=np.__version__),
        documentation=['https://www.statsmodels.org/stable/generated/statsmodels.stats.power.TTestPower.solve_power.html',
                       'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.nct.html'],
        inputs=[dict(path=p.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in inputs])
    return summaries,differences,meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--write',action='store_true');args=p.parse_args()
    summaries,differences,meta=calculate()
    if args.write:
        DEST.mkdir(parents=True,exist_ok=True)
        write_csv(DEST/'MDE9.csv',summaries);write_csv(DEST/'paired_task_differences225.csv',differences)
        (DEST/'method.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for name,computed in [('MDE9.csv',summaries),('paired_task_differences225.csv',differences)]:
        stored=read(DEST/name);assert len(stored)==len(computed)
        for actual,expected in zip(stored,computed):
            for key,value in expected.items():
                if isinstance(value,(float,int)):assert abs(float(actual[key])-value)<1e-10,(name,key)
                else:assert actual[key]==value,(name,key)
    saved=json.loads((DEST/'method.json').read_text(encoding='utf-8'))
    assert saved['inputs']==meta['inputs']
    primary=[r for r in summaries if r['contrast'] in ['M-R','G-R']]
    report=dict(status='passed',comparisons=9,paired_differences=225,task_count=25,
        primary_unadjusted_MDE_range=[min(r['MDE80_unadjusted'] for r in primary),max(r['MDE80_unadjusted'] for r in primary)],
        primary_conservative_MDE_range=[min(r['MDE80_conservative'] for r in primary),max(r['MDE80_conservative'] for r in primary)],
        original_tests_changed=False,new_API_calls=0,new_training_jobs=0)
    if args.write:(DEST/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':main()
