# Pro-LiGen: data, models and reproducible analysis

Version **v0.4.2** adds Zenodo archival and citation metadata. Scientific code, models, records and figure assets are unchanged from **v0.4.1**.

Version **v0.4.1** adds the [current submission supplement](extensions/20261110/README.md): the 750-position ChatDrug comparison, source server scripts, API reproducibility records, curation/chemical-form audits, saved docking cases and sensitivity analysis. Original project software is now licensed under MIT.

```sh
python reproduce/verify_submission_supplement.py --out NEW_SUBMISSION_CHECK
```

This repository supports property-constrained molecular editing with exemplar retrieval, program-computed verification, candidate retention and final QED selection. Version **v0.3.1** adds a complete output-novelty audit and runnable entry. Version **v0.3.0** added the three-condition comparisons in the current manuscript: ring count, LogP and topological polar surface area (TPSA). The original v0.2.0 records and scientific assets remain intact.

Start with [the current result-to-file map](docs/CURRENT_RESULTS_TO_FILES.md) or the [Chinese running guide](docs/README_中文.md). The [initial-release guide](docs/README_v020.md) and its original commands remain available.

## Reproduce current-paper results

Use Python 3.12 with the numerical dependencies, from any working directory:

```sh
python -m pip install -r reproduce/requirements-current.txt
python reproduce/reproduce_current_paper.py --out my_current_results
python reproduce/verify_extended_selection.py --out my_selection_check
```

The first command verifies initial A/B/C records and the extended comparisons. The second verifies all 1,500 final choices exactly from lossless candidate exports. Output directories must be new; frozen inputs are not overwritten. Neither command accesses an API, trains a model or runs docking. Optional `--figures` recreates the quantitative figures with the dependencies in `reproduce/requirements-optional.txt`.

## Audit molecular and scaffold novelty

See [the novelty guide](docs/NOVELTY_AUDIT.md). Run `python reproduce/audit_novelty.py --source . --out NEW_NOVELTY_RESULTS` with NumPy and RDKit. The audit includes full-library and GIN-input overlap, uniqueness, nearest neighbors, Murcko scaffold novelty and representation sensitivities.

## Reselect references and validate training inputs

```sh
python -m pip install -r reproduce/requirements-gin.txt
python reproduce/retrieve_property_references.py --out my_property_retrieval
python reproduce/retrieve_property_references.py --recompute-library --out my_fresh_property_retrieval
python reproduce/train_property_adaptations.py --out my_training_input_check
```

These commands use the actual LogP/TPSA checkpoints, the 24,693-molecule single-component library, saved embeddings and reference inputs. The default retrieval run re-encodes query molecules; the second also re-encodes the entire library. The validation command checks all training/validation triplets and a forward loss without changing model weights. Optional `--run-training` runs the archived trainer in a new output directory; it is not part of default verification. The original ring encoder and its separate retrieval entry point remain unchanged.

See [GIN details](docs/GIN_EXTENDED_REPRODUCTION.md) for exact identities, float32 near-tie reporting and training inputs. Scientific-field exports include all retained proposed SMILES and computed values. Whole literature-containing dialogues, credentials, manuscript files and editable schematic artwork are not distributed.

## Cohorts and files

| Cohort | Purpose | Planned runs |
|---|---|---:|
| Study A | Four condition-guidance strategies |500|
| Initial ring B | T/M/G; retained as initial validation |375|
| Study C | Two continuations of 75 shared initial-ring histories |150|
| Extended Study B | T/M/G/R under ring, LogP and TPSA conditions |1,500|

Each extended run has two rounds of three proposal positions and at most one final molecule;1,469 runs deliver an eligible output. Planned runs and delivered records are not counts of unique molecular structures. The current docking subset uses the same 11 calibrated targets, with 660 generated-output positions and 99 deterministic library-return observations.

`extensions/20261102/data/` holds numerical exports; `gin/` contains weights, embeddings, triplets, training logs and original implementation; `lossless/` contains completed scientific records; `scoring/` contains actual receptors, preparation identities, poses and scoring records. Original computational scripts and additional initial-cohort sensitivity analyses are archived separately. [Execution scope](docs/CURRENT_EXECUTION_SCOPE.md) describes each entry point. [TERMS.md](TERMS.md) records access and attribution; no new license is assigned to third-party material.
