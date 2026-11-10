"""Verify and package this independent run without credentials or unrelated project files."""
from pathlib import Path
import hashlib,json,re,sqlite3,zipfile
from common import ROOT,read,dump,sha,now
def main():
 assert read(ROOT/'analysis/full_audit.json')['results']==750
 assert read(ROOT/'analysis/final_summary.json')['formal_positions']==750
 assert read(ROOT/'logs/generation_status.json')['completed']==750
 for base in [ROOT,ROOT/'pilot']:
  db=sqlite3.connect(base/'logs/usage.sqlite');assert not db.execute("SELECT COUNT(*) FROM calls WHERE state='started'").fetchone()[0]
  db.execute('PRAGMA wal_checkpoint(TRUNCATE)');db.close()
 dump(ROOT/'analysis/figure_data_sources.json',{
  'figures/native_comparison.png':dict(inputs={n:sha(ROOT/n) for n in ['analysis/native_summary6.csv','analysis/paired_tests6.csv']},script='code/analyze.py',script_sha256=sha(ROOT/'code/analyze.py')),
  'figures/common_selector_frontier.png':dict(inputs={'analysis/candidate_frontier_summary.csv':sha(ROOT/'analysis/candidate_frontier_summary.csv')},script='code/analyze.py',script_sha256=sha(ROOT/'code/analyze.py'))})
 runtime_only={'logs/completion_status.json','logs/finisher.log','logs/package.py.log','completion.json'}
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.tmp','.lock'] and not p.name.endswith(('-wal','-shm')) and p.name!='file_manifest.json' and p.relative_to(ROOT).as_posix() not in runtime_only]
 secret_files=[]
 for p in files:
  if p.suffix.lower() in ['.py','.json','.jsonl','.csv','.md','.txt','.log']:
   text=p.read_text(encoding='utf-8',errors='replace')
   if re.search(r'\bsk-[A-Za-z0-9_-]{16,}\b',text):secret_files.append(p.relative_to(ROOT).as_posix())
 assert not secret_files,'Possible token found; inspect file names only: '+str(secret_files)
 manifest={p.relative_to(ROOT).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(files)}
 dump(ROOT/'file_manifest.json',dict(created_utc=now(),files=manifest,formal_positions=750,separate_pilot_positions=12,credential_scan_passed=True))
 out=ROOT.parent/'ProLiGen_ChatDrug_750_20261108_results.zip'
 with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files+[ROOT/'file_manifest.json']:z.write(p,ROOT.name+'/'+p.relative_to(ROOT).as_posix())
 with zipfile.ZipFile(out) as z:assert z.testzip() is None
 receipt=dict(status='complete',archive=str(out),bytes=out.stat().st_size,sha256=sha(out),files=len(files)+1,
  formal_positions=750,separate_pilot_positions=12,manuscript_edited=False,publicly_uploaded=False,finished_utc=now())
 dump(ROOT/'completion.json',receipt);print(json.dumps(receipt))
if __name__=='__main__':main()
