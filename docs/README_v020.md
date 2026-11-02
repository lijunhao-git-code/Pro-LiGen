# Pro-LiGen: research data and analysis code

This repository provides the retained molecular records and executable analysis supporting the Pro-LiGen study of conditional molecular generation with exemplar retrieval and programmatic verification. The evaluated edit is exactly one fewer RDKit graph ring than the seed, with single-component and noncopy requirements.

The release contains all final-position records used for Studies A, B and C, the retained candidate records needed to replay final selection, the reference molecules, recalibrated docking scores and receptor checks, quantitative plotting inputs, and analysis/plotting code. It also includes the deployed GIN checkpoint, complete inference source, saved library embeddings, the 26,108-entry retrieval library, and retained training triplets, curve and log. No API access or server login is needed for the numerical analysis.

Start with the [file and source map](docs/SOURCE_MAP.md), or the [Chinese reading and running guide](docs/README_中文.md). Original server files remain unchanged; the published copies and projections are indexed with their source hashes.

## Reproduce the reported numerical results

Use Python 3.12 in a virtual environment. From this repository root:

```sh
python -m pip install -r reproduce/requirements-analysis.txt
python reproduce/offline_analysis.py --out reanalysis_run_01
```

The command recomputes summaries from row-level observations and checks them against `reproduce/expected_frozen_results.json`. It writes `recomputed_results.json` and `verification.json` into a NEW directory. It does not overwrite records, call a model, train an encoder or run docking. Target means are the statistical units; the original Holm test families are preserved.

Optional chemistry checks and quantitative figure rendering:

```sh
python -m pip install -r reproduce/requirements-optional.txt
python reproduce/offline_analysis.py --out reanalysis_run_chemistry --chemistry
python reproduce/render_figures.py --out regenerated_figures_01
```

See [the result-to-file index](docs/RESULTS_TO_FILES.md) for every main table and result figure, [the data dictionary](docs/DATA_DICTIONARY.md), and [the docking protocol](docs/DOCKING_PROTOCOL.md). JSON records are authoritative; `data/csv` contains convenient CSV mirrors. Existing figure assets are the paper's actual saved quantitative figures.

## Reproduce Morgan and frozen-GIN reference selection

Version **v0.2.0** adds the complete frozen-GIN graph builder and encoder, restored from the retained experiment sources, and an independent retrieval entry point. It reads only released files and runs on CPU. No server login, language-model API key, training or docking is required.

```sh
python -m pip install -r reproduce/requirements-gin.txt
python reproduce/retrieve_references.py --out gin_run_cached
python reproduce/retrieve_references.py --recompute-library --out gin_run_full
```

The first command recomputes the seed embeddings using the released checkpoint and uses the saved library embeddings. The second also rebuilds and encodes all **26,108** library molecules from SMILES. Both compare the ordered top-three references for all **25** tasks and both backends against the original **150** reference records. Outputs include selected SMILES, similarities, per-task checks, software versions and input hashes. The exact validated Python 3.10 environment is listed in `reproduce/requirements-gin-validated.txt`; the broader requirements file also supports newer Python environments.

See [the GIN reproduction guide](docs/GIN_REPRODUCTION.md) for the single-task check, output interpretation and source provenance. Core numerical analysis and figure rendering remain available through the earlier commands; their original Python 3.12 dependency pins are separate from the validated GIN environment.

## Inspect and validate the retained training material

```sh
python reproduce/verify_training_inputs.py --out gin_run_training_inputs
```

This checks all **15,461** retained training triplets and their ring-count labels, the **26,402** original SMILES rows and their agreement with the deduplicated library, the **50** recorded training epochs, and function identity against the original deployed source. It does not train a model. The original trainer, triplet builder and their inputs are indexed in [SOURCE_MAP.md](docs/SOURCE_MAP.md).

## Included studies

| Study | Purpose | Released final records |
| --- | --- | --- |
| A | Compare four guidance strategies across 25 target-seed tasks | 500 retained outputs |
| B | Compare no examples (T), Morgan examples (M) and frozen-GIN examples (G) | 375 planned positions, including 2 nondeliveries |
| C | Compare additional computed feedback with a matched continuation | 150 branches from 75 shared histories |

Each B run can propose up to three molecules in each of two rounds, but returns at most one final molecule. The 375 planned positions are not the number of all proposed molecules. C retains the same starting candidates in both branches. Docking records cover 900 observations, including seeds, exclusions and missing outputs; the primary calibrated nonmetal comparison uses 11 targets.

## Release scope

Runnable entry points cover numerical result reproduction, quantitative figure rendering, and Morgan/frozen-GIN reference selection, including library re-encoding. The original encoder snapshot is preserved under `models/`; the executable implementation is `proligen/gin.py`, and the retained training implementation is `archive/train_gin.py`.

Complete literature-containing API dialogues, credentials, account records, manuscript files and editable schematic artwork are outside this release. Historical generation/docking scripts remain archived implementation records with their original path assumptions. The exact contents and execution scope are described in [RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md). See [TERMS.md](TERMS.md) for reuse terms and third-party attribution.
