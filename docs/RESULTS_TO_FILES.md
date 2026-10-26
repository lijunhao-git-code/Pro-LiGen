# Reported results and their sources

This index follows the current three-result-table manuscript with GIN Figures 2 and 3 retained. Internal asset names preserve their historical numbering.

| Paper item | Primary observations | Executable analysis / frozen plotting input | Released display |
| --- | --- | --- | --- |
| Table 1, Study A guidance strategies | `data/SI_records/A_retained500.json` | `historical()` in `reproduce/offline_analysis.py` | `paper_tables/table1_historical.tex` |
| Table 2, exemplar comparison and paired tests | `B_planned375.json` | `current()`, `paired()`, `correct_holm()` | `paper_tables/table3_downstream.tex` |
| Table 3, Study B docking | `docking_observations900.json` | `docking()` | `paper_tables/table4_docking.tex` |
| Figure 4, Study A trade-offs | `A_retained500.json` and `A_target_means100.json` | `A_targets` in `reproduce/figure_inputs_frozen.json`; `history()` in `audit/b05_make_figures.py` | `figures/fig2_historical_tradeoffs.*` |
| Figure 5, target-paired exemplar effects | `B_planned375.json` and `B_target_means75.json` | `facts`, `B_targets`; `downstream()` | `figures/fig4_downstream.*` |
| Figure 6, docking/QED trade-off | `docking_observations900.json` | `dock_target`; `docking()` in the figure renderer | `figures/fig5_docking_tradeoffs.*` |
| Study C and Table S10 | `C_branches150.json`, `C_completed_branches150_with_candidates.json` | `feedback()`; frozen `C_targets` | `figures/fig3_mechanism.*`, panel a |
| Candidate replay, Table S5 | `B_candidates_lossless2244.json` in `reproduce/`, `B_fixed_candidate_replay1500.json` | `choose()`, `replay_and_selection()` | `figures/fig3_mechanism.*`, panel b |
| Reference controls | `B_references150.json` | frozen `reference_rows`; `reference()` | `figures/figS2_reference_controls.*` |
| Fixed chemical illustration | `tasks25.json`, `B_planned375.json`, `B_references150.json` | frozen `case_rule`, `case_molecules`; `chemistry()` | `figures/figS3_structural_example.*` |
| Full A/B docking, Table S11 | `docking_observations900.json` | seven group summaries in `docking()` | recomputed JSON, `docking/groups` |

Unless stated otherwise, JSON files above are under `data/SI_records/`. The numerical entry point produces all table group summaries and the complete original B/C statistics, including delivery-rate tests omitted from the compact main table. Main Figure 1 and the GIN schematic Figures 2–3 are explanatory drawings, not plots of measurements, and are outside this results release.

Manual check: find a target by `chemblid`, match `group` and position/slot, then compare its SMILES, eligibility, QED and SA. For a table mean, group rows by target first, then average the target means. For docking compare each output against its same-protocol seed; a positive output-minus-seed Vina difference is worse. Do not replace missing values with zero, except for the defined planned-position QED utility.
