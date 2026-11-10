# Submission supplement, 10 November 2026

This additive extension provides the current manuscript's Study D, curation and chemical-form audits, API settings/templates and measured resource records, saved docking-pose cases, and paired-t sensitivity benchmarks. Earlier A-C, three-property GIN adaptations, training inputs and checkpoints remain at their existing repository paths.

Run from the repository root:

```sh
python reproduce/verify_submission_supplement.py --out NEW_SUBMISSION_CHECK
```

Python 3.10+ and NumPy are sufficient for the four default checks. These checks replay frozen observations and verify source/figure hashes; they make no API calls, do not retrain models, and do not rerun docking.

| Evidence | Scientific records | Offline entry |
|---|---|---|
| Main Table 3, external-method comparison | `data/external20261108/`: 750 outputs, 4,390 candidate positions, task means, paired tests, common-selection replay | `reproduce/external_comparison.py` |
| API settings, complete prompt templates, recorded usage | `data/reproducibility20261107/` | `data/reproducibility20261107/reproduce_api_resources.py` |
| SI S15, input curation and overlaps | `data/curation20261102/` | `reproduce/audit_data_curation.py --source REPOSITORY_ROOT --out NEW_DIRECTORY` (RDKit required) |
| SI docking eligibility and chemical-form mapping | `data/docking_reporting20261103/` | Frozen per-task records and mappings |
| SI S18 / Figure S8, two saved-pose cases | `data/docking_cases20261108/` | `reproduce/verify_docking_cases.py` |
| SI S12.3, minimum detectable effects | `data/power20261109/` and the three exact source tables in `data/updated20261102/` | `reproduce/power_sensitivity.py` |

The current main docking comparison is Table 4 (the historical filenames in the 20261102 extension remain unchanged). The external-method comparison is an independent 750-position cohort; it is not an extra GIN-vs-Morgan arm.

`archive/chatdrug_code/` and `archive/docking_code/` contain the actual server scripts, copied byte-for-byte on 10 November. The server inventory records their hashes. These are archived launch implementations with their original server paths; portable numerical verification uses the entries above. The external-data package also includes original-precision scientific results, source hashes and comparison-specific prompt construction. Full third-party literature dialogues, account files and API secrets are outside the distribution. No default command invokes the API.

Original project software and documentation are covered by the root MIT LICENSE. Third-party code, source datasets, model inputs and receptor structures retain their own terms and attribution; see TERMS.md. The ChatDrug adaptation cites its upstream repository/commit in `data/external20261108/upstream_source.json`; it is not presented as the authors' original algorithm.
