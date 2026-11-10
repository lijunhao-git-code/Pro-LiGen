# Saved-pose cases, 2026-11-08

Two existing Study-B G trajectories, selected by a frozen median-score rule before contact inspection. This is an illustrative post-analysis with zero new generation calls and zero new docking calculations. Main Table 4 and Figure 6 keep their original numerical inputs.

Run `python reproduce/verify_docking_cases.py` from the manuscript root for standard-library-only verification. Full reanalysis: `python data/docking_cases20261108/scripts/analyze_docking_pose_cases_20261108.py --root data/docking_cases20261108`, then the analogous plotting script. Dependencies: RDKit, Meeko 0.7.1, SciPy, NumPy, PLIP 3.0.1, Open Babel 3.1.1, PyMOL 3.2.0a0, Matplotlib and Pillow. Third-party tools retain their own licenses and are not bundled.

`inputs/selection_rule.json` records the fixed rule; `selection_sources/` contains the original task and output summaries. `inputs/input_manifest.json` maps original PDBQT/SDF/PDB/scoring files to v0.3.1 release paths and hashes. `analysis/cases.json` gives the checked case numbers. `analysis/all_interactions.csv` preserves all 44 PLIP default records. `analysis/figure_plotted_contacts.csv` records each displayed mark; short H-bonds additionally require D-A <=3.5 Angstrom and donor angle >=120 degrees. `analysis/verification.json` records identity and coordinate checks.

The original pose is reconstructed without fitting or minimization. Polar-hydrogen coordinates from the initial PLIP/Open Babel pass are frozen as `frozen_annotation_complex.pdb`, then reused with further hydrogen addition disabled for repeatable annotation; no scores are recalculated. Raw tertiary-amine salt-bridge annotations are retained in the full PLIP export, while the figure displays hydrophobic, aromatic and screened H-bond contacts. Cases illustrate geometry, not experimentally measured affinities. The TPSA case has a negative QED change, retained under the numerical selection rule.

PyMOL `.pse` scenes and camera records permit graphical inspection. The four predicted first poses and protein--ligand complexes are provided as PDB/SDF. All original scoring records are preserved.
