# Current manuscript results to files (v0.3.1)

All paths below are relative to the repository root. Display numbers refer to the current manuscript, not historical script filenames.

| Display | Frozen records | Reproduction |
|---|---|---|
| Table 1 / Figure 4, Study A | `data/SI_records/A_retained500.json`; original `reproduce/figure_inputs_frozen.json` | `reproduce/offline_analysis.py`; `reproduce/render_figures.py` (historical trade-offs) |
| Table 2, three conditions × T/M/G/R | `extensions/20261102/data/updated20261102/observations1500.csv`, `table2_values.csv`, source final-position exports | `reproduce/reproduce_current_paper.py`; extended `reproduce/extended_comparison.py` |
| Figure 5, task differences and9 contrasts | Same directory: `figure5_target_differences75.csv`, `figure5_comparisons9.csv`, actual plotted coordinates | Extended `reproduce/plot_extended_comparison.py` |
| Table 4 / Figure 6, same 11 tasks | `extensions/20261102/data/mechanism_docking20261102/docking_observations759.csv`, `docking_target_means231.csv`, `table3_values12.csv`, figure coordinates | Extended `reproduce/mechanism_docking.py`, `plot_mechanism_docking.py` |
| Study C, added computed feedback | Initial `data/SI_records/C_*` and frozen expected results | `reproduce/offline_analysis.py`; original primary p value retained |
| First/random/highest-QED final choice and46 retained positions | Extended candidate grids, `selection_replay7500.csv`, round availability; `extensions/20261102/lossless/*_completed.json` | Numerical entry point plus `reproduce/verify_extended_selection.py` |
| Extended reference pool and distributions | `extensions/20261102/gin/inputs/library.csv`, `references2250.csv`, `reference_pools50.csv`; ring references in extended data | `reproduce/retrieve_property_references.py`; original `reproduce/retrieve_references.py` for ring M/G; ring R records/seeds in protocol |
| GIN property training | `extensions/20261102/gin/models/`, `training/*_triplets.csv`, `*_epochs.csv`, `*_status.json`, original code | `reproduce/train_property_adaptations.py`; checkpoints selected by validation loss |
| Actual scoring evidence | `extensions/20261102/scoring/{ring,properties}/jobs/`, receptor files, protocol | Observation/job identifiers link raw scores and poses; numeric verification does not rerun docking |
| Novelty paragraph / SI Tables S19--S21 | `extensions/novelty20261102/results/` and unchanged released source library/triplets/lossless records | `reproduce/audit_novelty.py`; [definitions](NOVELTY_AUDIT.md) |
| Initial-ring MCS/threshold and selection diagnostics | `extensions/20261102/archive/initial_structural_sensitivity`, `initial_selection_diagnostics` | Original code and full numerical exports retained with initial-cohort labels |

The ring, continuous-property and initial cohorts retain separate identities. Conditional descriptors first average successful observations within a task, then across tasks; S/U keep the planned-run denominator. Current manuscript table snippets and quantitative figure assets are in `extensions/20261102/paper_tables/` and `figures/`. Their filenames retain the original project naming convention.

## Current submission additions (v0.4.0)

Main Table 3 is the 750-position external-method comparison, not the docking table. See [the submission supplement](../extensions/20261110/README.md) for its exact records and all new SI analyses. The original docking table is now Table 4; its scientific records and observation definitions are unchanged. Run `python reproduce/verify_submission_supplement.py --out NEW` for the new default offline checks.
