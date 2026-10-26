# Retained docking protocol and files

`docking/protocol.json` is the saved protocol, copied unchanged. It records Vina scoring with seed 42, exhaustiveness 16, 10 poses and one CPU per job. Ligands use the original formal charges, explicit hydrogens, RDKit embedding with seed 42 (random-coordinate fallback with seed 42), MMFF optimization up to 500 iterations, and Meeko preparation. No protonation-state enumeration was performed.

The scoring form is the unique carbon-containing component after removing atomic counterions; ambiguous multi-organic forms are not rescued. Parent records remain unchanged. Receptors and seeds are evaluated with the same protocol. Native cofactor handling, pocket centers/sizes, PDB source URLs, source/prepared hashes, calibration results and exclusions are available in `data/SI_records/receptor_grids25.json`, `receptor_review25.json` and `cocrystal_calibrations22.json`.

Selection is based on receptor review and top-ranked, in-place symmetry-aware heavy-atom redocking RMSD at most 2 Å, followed by metal-target exclusion. The main comparison uses 11 calibrated nonmetal targets. A positive generated-minus-seed Vina difference is worse. These are computational docking observations, not measured binding affinities.

`docking/receptors/`, `docking/calibration/` and `docking/docking/` contain the retained coordinate, ligand and pose files available in the exported experiment archive. Their directory names and contents are preserved. `archive/b02_docking.py` is the historical implementation, with its original server paths and guards; use it to inspect preparation/scoring rather than executing it unchanged on a different machine. Coordinates derive from the PDB sources identified in the receptor records.
