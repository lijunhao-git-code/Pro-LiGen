# Study D: Pro-LiGen versus ChatDrug adaptation

This is a separate contemporaneous external-method comparison, not an additional GIN-versus-Morgan arm. It contains 750 formal runs (25 tasks × three conditions × five repeats × two methods), excluding 12 technical-pilot positions. Original full experimental archive: `ProLiGen_ChatDrug_750_20261108_results.zip` (SHA-256 `cd9bcce45bef91ac0b08b307657ad92fdf7e201e92129162b45c9e7afe8e42e8`).

From the unpacked manuscript root run `python reproduce/external_comparison.py` (Python 3.10+, NumPy). This checks file hashes, the 750 scientific result records, 150 task means, all six main-table rows, six paired tests with Holm correction, 24 candidate-policy cells and 36 candidate-frontier cells. It does not contact a model API. Archived generation files are retained for inspection, not executed by the numerical verifier.

- Main Table 3: `native_summary6.csv` → `final_outputs750.csv` → `results/<slot_id>.json` → candidate identity and properties.
- Paired tests: `paired_tests6.csv` → `task_means150.csv` → the five output records per task/method/condition. Unit: 25 paired task means; one Holm family of six tests.
- Replay and supplementary figure: `candidate_replay_summary.csv` / `candidate_frontier_summary.csv` → `all_candidate_records.csv` (4,390 positions). Missing later proposals are not regenerated.
- Common inputs: `inputs/scientific_entries750.jsonl` contains seeds, numerical targets, G references, exact G task-stage prompts and context hashes; `inputs/pools75.json` contains the eligible pools. ChatDrug prompt templates and feedback/selection logic are in `archived_code/comparison.py` and `common.py`.
- Shared full-text literature background is not redistributed. Original request/response provenance and frozen input/code hashes are retained in `inputs/context_provenance.json` and `inputs/generation_locked.json`; the complete internal archive retains the transcripts. All new molecular proposals were generated in the external cohort.
- Native policies differ: Pro-LiGen uses QED-directed refinement and highest-QED selection; ChatDrug uses its first eligible parsed output. Findings concern complete workflows, not an isolated encoder effect or identical optimization objectives.
- `API_usage_by_call.csv` includes pilot and formal calls, explicitly labeled. Cost is an archived-rate-card estimate, not a recharge receipt. No docking or model training was added.

No GitHub upload or DOI assignment is performed by this integration.
