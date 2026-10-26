# Data dictionary and aggregation

- `chemblid`: target/task identifier, not a generated-molecule identifier. `uniprotid` identifies the supplied protein context.
- `slot_id`: planned generation/continuation position. `group` is an A strategy or B backend; `branch` identifies C feedback/no-extra-feedback continuations.
- `smiles` / `parent_smiles`: retained molecular representation. `canonical`: RDKit isomeric canonical SMILES. Multi-fragment references are allowed; qualified final B/C outputs must be single-component.
- `RC`: RDKit graph-ring count. `dRC_seed_minus_output` is seed RC minus output RC; the exact-one-less diagnostic equals one when this difference is exactly 1.
- `exact_noncopy_success`, `ok`: qualified final delivery. `compliant_QED_utility` / `yield_QED`: delivered QED, or zero if no final delivery. Failed molecular-property fields remain missing.
- `QED`: drug-likeness score, higher is better. `SA_raw` / `SA_standard_raw`: standard synthetic-accessibility score, lower is easier. Historical heuristic SA fields are retained for provenance and are not the current standard-SA result.
- `tanimoto` / historical `sim`: binary Morgan structural Tanimoto similarity. Reference `similarity` is the backend's cosine ranking score, with different representation scales.
- `turn` 4 and 5: the two proposal rounds after three contextual turns. `candidate_position` runs from 1 to 3. `eligible` is a proposal-level qualification flag; `reason` records failures.
- `docking_status`: whether an observation was scored or excluded. `vina`: saved Vina score in kcal/mol. `dVina_output_minus_seed`: positive means worse than the matched seed. `dQED_active_output_minus_seed`: QED change of the scored organic component.
- `parent_*` and `active_*`: full parent-form and scored organic-component properties, respectively. They must not be silently interchanged.

The primary B analysis uses 25 paired target means, not 375 independent samples. G−M has a two-test Holm family (delivery and QED utility); G−T and M−T share a four-test auxiliary family. Bootstrap intervals are unadjusted. C has a single primary paired test. Docking first averages scored observations within each target/group and then averages across 11 calibrated nonmetal targets.

The JSON records and full-precision candidate export preserve floating-point values. CSV mirrors are for inspection; use the lossless JSON for exact tie-sensitive candidate replay.
