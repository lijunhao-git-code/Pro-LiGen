"""Verify docking score origins and actual prepared structures, without redocking."""
import re
from rdkit import Chem,RDLogger
from common import ROOT,read,sha,now
from analysis_utils import save_json,csv


def main():
    assert read(ROOT/'docking/complete.json')['status']=='complete'
    RDLogger.DisableLog('rdApp.*');receptors=read(ROOT/'inputs/docking_receptors11.json')
    for r in receptors.values():assert sha(ROOT/r['path'])==r['sha256']
    rows=[]
    for folder in sorted((ROOT/'docking/jobs').iterdir()):
        if not folder.is_dir():continue
        result=read(folder/'result.json');task=read(folder/'task.json')
        assert task['canonical']==result['canonical'] and task['chemblid']==result['chemblid']
        row=dict(job_id=folder.name,chemblid=result['chemblid'],canonical=result['canonical'],status=result['status'],
            result_sha256=sha(folder/'result.json'),task_sha256=sha(folder/'task.json'))
        if result['status']=='ok':
            assert task['receptor']['sha256']==receptors[task['chemblid']]['sha256']
            mol=next(Chem.SDMolSupplier(str(folder/'prepared_ligand.sdf'),removeHs=True))
            assert mol is not None
            canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
            ref=Chem.MolFromSmiles(result['canonical']);plain_a=Chem.Mol(mol);plain_b=Chem.Mol(ref)
            Chem.RemoveStereochemistry(plain_a);Chem.RemoveStereochemistry(plain_b)
            assert Chem.MolToSmiles(plain_a)==Chem.MolToSmiles(plain_b),folder.name
            assert mol.GetNumAtoms()==ref.GetNumAtoms() and mol.HasSubstructMatch(ref,useChirality=True),folder.name
            scores=re.findall(r'^REMARK VINA RESULT:\s*([-+0-9.]+)',(folder/'poses.pdbqt').read_text(),flags=re.M)
            assert scores and abs(float(scores[0])-result['vina'])<=.0011
            assert abs(result['energies'][0][0]-result['vina'])<1e-12
            row.update(vina=result['vina'],pose_first_score=float(scores[0]),prepared_ligand_connectivity_and_specified_stereo_match=True,
                prepared_SDF_canonical=canonical,prepared_stereo_SMILES_exact_match=canonical==result['canonical'],
                SDF_sha256=sha(folder/'prepared_ligand.sdf'),ligand_PDBQT_sha256=sha(folder/'ligand.pdbqt'),poses_sha256=sha(folder/'poses.pdbqt'))
        rows.append(row)
    csv(ROOT/'analysis/docking/docking_source_audit.csv',rows)
    out=dict(status='passed',time_utc=now(),unique_jobs=len(rows),successful_jobs=sum(r['status']=='ok' for r in rows),
        all_successful_prepared_SDF_connectivities_and_input_specified_stereo_match=True,
        all_successful_first_pose_scores_match_saved_Vina_and_energy_matrix=True,
        all11_receptor_hashes_match=True,new_API_calls=0,new_docking_jobs=0)
    save_json(ROOT/'analysis/docking/docking_source_verification.json',out);print('DOCKING SOURCE AUDIT',out,flush=True)


if __name__=='__main__':main()
