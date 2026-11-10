"""No-API behavioral checks for the adapted baseline and durable request accounting."""
import copy,json,tempfile
from pathlib import Path
from rdkit import Chem,DataStructs,RDLogger
from rdkit.Chem import AllChem
from common import ROOT,read,read_jsonl,dump,properties,evaluate_smiles,choose
from comparison import ComparisonRunner,Retrieval,native_first,occurrences

def main():
 RDLogger.DisableLog('rdApp.*');entries=read_jsonl(ROOT/'inputs/entries750.jsonl');pools=read(ROOT/'inputs/pools75.json')
 e=next(e for e in entries if e['property']=='logp' and e['group']=='CD')
 examples=[]
 for s in pools[e['prefix_key']][:200]:
  row=evaluate_smiles(s,e,4,len(examples)+1)
  if row['eligible']:examples.append(row)
  if len(examples)==15:break
 assert len(examples)>=2
 examples.sort(key=lambda x:x['QED']);a,b=examples[0],examples[-1];assert a['QED']<b['QED']
 assert native_first('- '+a['smiles']+'\n- '+b['smiles'],e['seed_smiles'])==a['smiles']
 assert native_first('- '+e['seed_smiles']+'\n- '+a['smiles'],e['seed_smiles'])==a['smiles']
 r=ComparisonRunner.__new__(ComparisonRunner);r.contexts={e['prefix_key']:[dict(role='system',content='test')]}
 class Retrieve:
  def closest(self,key,failed):return a['smiles'],.5
 r.retrieval=Retrieve()
 slot=dict(slot_id='mock',group='CD');calls=[]
 with tempfile.TemporaryDirectory(dir=ROOT/'logs') as td:
  r.root=Path(td)
  def reply(*args):calls.append(args);return '- '+a['smiles']+'\n- '+b['smiles']
  r.reply=reply
  selected,rows,refs,trace,status=r.execute(slot,e,1,{})
  assert selected['smiles']==a['smiles'] and len(calls)==1 and len(rows)==2
  assert choose(rows)['smiles']==b['smiles'],'Common replay should differ from the native first success when QED differs'
  failed='CCCCCCCCCCCCCCCCCCCC';assert not evaluate_smiles(failed,e)['property_hit']
  r.reply=lambda *args:failed;selected,rows,refs,trace,status=r.execute(slot,e,1,{})
  assert selected is None and len(trace)==3 and len(refs)==2 and status=='native_round_limit'
  r.reply=lambda *args:'CCCCCC(';selected,rows,refs,trace,status=r.execute(slot,e,1,{})
  assert selected is None and status=='native_invalid' and len(trace)==1
  r.reply=lambda *args:'No candidate';selected,rows,refs,trace,status=r.execute(slot,e,1,{})
  assert selected is None and status=='native_unparsed'
 retrieval=Retrieval();seq,fps=retrieval.get(e['prefix_key']);query='CCCCCCCCCCCCCCCCCCCC';ref,score=retrieval.closest(e['prefix_key'],query)
 scores=DataStructs.BulkTanimotoSimilarity(AllChem.GetMorganFingerprint(Chem.MolFromSmiles(query),2),fps)
 assert score==max(scores) and ref==seq[scores.index(max(scores))]
 # Exact range endpoints and copy checks are distinct from property qualification.
 copyrow=evaluate_smiles(a['smiles'],dict(e,exemplars=[a['canonical']]))
 assert copyrow['property_hit'] and not copyrow['eligible'] and copyrow['copy_exemplar']
 ring=next(x for x in entries if x['property']=='ring' and x['group']=='G')
 assert properties(ring['seed_smiles'])['RC']-1==ring['target_center']
 for ent in entries:
  assert ent['tolerance']>=0 and ent['target_lower']<=ent['target_center']<=ent['target_upper']
  assert ent['group']!='G' or len(ent['exemplars'])==3
 pairs={}
 for ent in entries:pairs.setdefault((ent['property'],ent['chemblid'],ent['position']),[]).append(ent)
 assert len(pairs)==375 and all(len(pair)==2 and pair[0]['prefix_sha256']==pair[1]['prefix_sha256'] for pair in pairs.values())
 dump(ROOT/'inputs/runner_self_tests.json',dict(status='passed',paid_calls=0,checks=[
  'native first candidate is not highest-QED candidate','literal seed exclusion','first eligible stop','C2 means at most three proposal replies',
  'invalid and unparsed are scientific failures without quality retries','nearest sparse-count Morgan retrieval from failed candidate',
  'copy rejection independent of property hit','ring exact minus1','375 paired inputs and identical frozen context','shared selection replay distinguishes native choice']))
 print(json.dumps(dict(status='passed',paid_calls=0,paired_inputs=375)))
if __name__=='__main__':main()
