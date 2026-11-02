# Current three-condition novelty audit

```sh
python -m pip install -r reproduce/requirements-novelty.txt
python reproduce/audit_novelty.py --source . --out NEW_NOVELTY_RESULTS
```

This entry does not access a model API, train, dock, or modify frozen inputs. It checks all 1,500 planned runs, 1,469 delivered records and 1,069 distinct molecules against the complete 26,108-molecule library and all released GIN triplet participants. Ring uses the archived RC input union; LogP/TPSA training and validation participants are also evaluated separately. Exact input membership, stereo-insensitive and atomic-component sensitivities, whole-library/GIN nearest neighbors, typed Murcko scaffolds and generic frameworks are exported.

Frozen results: `extensions/novelty20261102/results/`. `protocol_and_sources.json` defines representations and denominators. `final_records1500.csv` identifies every result and source hash; `unique_molecule_audit.csv` gives the closest reference structure and identifier; `reference_overlap_matrix108.csv` checks each configuration against each collection. `group_summary12.csv` supplies the new SI tables. The code automatically compares recomputed CSV identities and numerical values with these frozen results.

One ring-G record exactly matches the library and archived GIN inputs. Removing stereochemical specification adds one ring-R match. These are distinct from the per-run exclusion of seed and supplied-example copies. No structure or original success flag is changed. Novelty refers to the explicitly named reference collections; own-seed scaffold retention and library-relative novelty are reported separately.
