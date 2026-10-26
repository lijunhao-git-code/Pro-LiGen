# Reproduce the frozen GIN and Morgan reference sets

The original Study-B checkpoint and retrieval inputs are retained unchanged. This entry point reconstructs graph inputs from SMILES and selects the three references per task/backend. It does not train GIN or call a language-model service.

## Environment

Install PyTorch for your platform, then install `reproduce/requirements-gin.txt` in an isolated environment. A CPU installation is sufficient; optional PyG CUDA/scatter extensions are not required by this entry point. For PyTorch's CPU wheels, use the official PyTorch installation channel appropriate to your platform.

The exact validated configuration uses Python 3.10, with pins in `reproduce/requirements-gin-validated.txt`. Actual runtime versions are also included in each verification report. The original numerical-analysis environment uses Python 3.12 and its existing dependency pins; the GIN command does not import pandas or matplotlib.

## Commands

From the repository root, first check one task if desired:

```sh
python reproduce/retrieve_references.py --task CHEMBL1741186 --out gin_run_smoke
```

Reproduce both reference sets for all 25 tasks using the saved library embeddings:

```sh
python reproduce/retrieve_references.py --out gin_run_cached
```

Recompute the entire library from SMILES and the released checkpoint:

```sh
python reproduce/retrieve_references.py --recompute-library --out gin_run_full
```

The output directory must be new. Paths to all inputs are resolved from the repository location, independently of the shell's current directory. Library encoding is batched to limit memory use; `--batch-size` defaults to 128 and `--threads` to 4. Both commands use CPU and never overwrite the checkpoint, library, saved embeddings or reference observations.

## What is checked

- 26,108 unique canonical library structures and 256-dimensional embeddings.
- The checkpoint SHA256 equals the frozen Study-B checkpoint.
- Morgan uses radius-2, 2,048-bit fingerprints and binary cosine ranking.
- Both backends filter to the same seed-relative exact-ring pool before ranking and exclude canonical seed copies.
- Descending cosine scores with ascending canonical SMILES for exact ties select three references.
- The ordered reference identities match `data/SI_records/B_references150.json` for 25 tasks and both backends: 150 records total.
- Selected cosine scores agree within absolute tolerance `1e-6`. In full-library mode, the recomputed embedding array is compared with the saved array using the same absolute tolerance.
- Input hashes remain unchanged.

`references.json` and `references.csv` contain the recomputed references, pool conditions, identities and cosine scores. `verification.json` records per-task agreement, observed numerical differences, software versions and input hashes. A failed check produces a nonzero exit status and preserves the diagnostic report; it does not change the stored observations.

## Code provenance

`proligen/gin.py` includes all four required definitions. `atom_features` and `bond_features` are extracted unchanged from the retained `archive/train_gin.py`; `mol_to_graph` and `GINEncoder` are the exact functions saved in the Study-B `models/encoder_source.json` snapshot. Function-level AST identity and source hashes are recorded in `provenance/gin_source_restoration.json`.

The separate 2026-10-12 audit already reproduced the top-three GIN references on all 25 tasks, with a maximum recorded CPU-versus-saved score difference of about `1.79e-7`. The v0.2.0 reports under `verification/gin_cached_library.json` and `verification/gin_recomputed_library.json` validate the newly packaged entry point. These are execution checks of the retained model, not additional molecular-generation experiments.
