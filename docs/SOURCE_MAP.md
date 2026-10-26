# File organization and original sources

The default commands are under `reproduce/`. Reusable inference code is under `proligen/`. `data/` contains scientific inputs and records; `models/` contains the deployed checkpoint. `archive/` preserves original implementations and is not the default working directory.

Original server files were read without modification. Paths in this table are relative to the original experiment project. Byte-identical copies are listed in `provenance/server_source_index.json`; the SMILES projection is documented separately in `provenance/training_smiles_projection.json`.

| Purpose | Published file(s) | Original source / relationship |
| --- | --- | --- |
| Deployed GIN definitions | `archive/original_project/run.py`, `proligen/gin.py` | Original `run.py`; the inference module extracts four definitions unchanged |
| GIN trainer | `archive/train_gin.py` | `mw_experiments/train_gin.py` |
| Original triplet builder | `archive/build_triplets.py` | `mw_experiments/build_triplets.py` |
| Training triplets | `data/training/triplets_ring_count.csv` | Same file under `mw_experiments/triplets/`; 15,461 rows |
| Training curve and log | `data/training/best_gin_rc.csv`, `train_rc.log.txt` | `mw_experiments/models/best_gin_rc.csv`, `train_rc.log`; 50 retained epochs |
| Source molecular inputs | `data/training/source_smiles26402.csv` | Select only `smiles` from `csv_folder/train.csv`; original order and duplicates retained |
| Frozen inference inputs | `models/best_gin_rc.pth`, `data/retrieval/*` | Study-B checkpoint, saved normalized embeddings and canonical library |
| Target/seed source mapping | `data/task_sources/best_ligand.csv` | `csv_folder/best_ligand.csv`; 908 retained identifier/SMILES mappings |
| Study-A original outputs | `data/raw_outputs/study_a/exp30rc25_*.txt` | `mw_experiments/generated/`; four original final-output files |
| Study-A historical implementation | `archive/study_a/*`, `archive/retrieve_rc.py` | Original prompt builders, generation and retrieval scripts |
| Study-B implementation/protocol | `archive/generation/*`, `data/protocols/study_b_original.json` | Original 375-position experiment scripts and protocol |
| A/B/C row records and candidates | `data/SI_records/*` | Retained scientific exports; sources/checksums in `source_provenance.json` |
| Receptor/docking records and coordinates | `data/SI_records/receptor_*.json`, `docking/` | Reviewed/calibrated scoring inputs and retained coordinate/pose files |
| Result summaries and figures | `reproduce/`, `figures/`, `paper_tables/` | Quantitative reproduction entry points, frozen inputs and saved displays |

## Runnable entry points

| Command | What it does | Trains / calls a model API / runs docking? |
| --- | --- | --- |
| `python reproduce/offline_analysis.py --out reanalysis_run_01` | Recompute A/B/C summaries, tests, saved-candidate selections and docking summaries | No |
| `python reproduce/render_figures.py --out regenerated_figures_01` | Render quantitative figures from frozen inputs | No |
| `python reproduce/retrieve_references.py --out gin_run_cached` | Recompute query embeddings and reference selection using cached library embeddings | No |
| Same command with `--recompute-library` | Also encode all 26,108 library molecules from SMILES | No |
| `python reproduce/verify_training_inputs.py --out gin_run_training_inputs` | Check training artifacts, ring-count labels, molecular source/library agreement and code identity | No |

Optional training uses `archive/train_gin.py` with explicit input/output arguments, as documented in `GIN_REPRODUCTION.md`. The archived original `run.py` contains example paths; use the portable retrieval entry point for the released benchmark. Historic A, B and calibrated docking protocols remain distinct, and their files are kept separate.
