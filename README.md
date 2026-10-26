# Pro-LiGen: research data and analysis code

This repository provides the retained molecular records and executable analysis supporting the Pro-LiGen study of conditional molecular generation with exemplar retrieval and programmatic verification. The evaluated edit is exactly one fewer RDKit graph ring than the seed, with single-component and noncopy requirements.

The release contains all final-position records used for Studies A, B and C, the retained candidate records needed to replay final selection, the reference molecules, recalibrated docking scores and receptor checks, quantitative plotting inputs, and analysis/plotting code. It also includes the deployed GIN checkpoint, its saved library embeddings and the 26,108-entry retrieval library. No API access or server login is needed for the numerical analysis.

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

## Included studies

| Study | Purpose | Released final records |
| --- | --- | --- |
| A | Compare four guidance strategies across 25 target-seed tasks | 500 retained outputs |
| B | Compare no examples (T), Morgan examples (M) and frozen-GIN examples (G) | 375 planned positions, including 2 nondeliveries |
| C | Compare additional computed feedback with a matched continuation | 150 branches from 75 shared histories |

Each B run can propose up to three molecules in each of two rounds, but returns at most one final molecule. The 375 planned positions are not the number of all proposed molecules. C retains the same starting candidates in both branches. Docking records cover 900 observations, including seeds, exclusions and missing outputs; the primary calibrated nonmetal comparison uses 11 targets.

## Release scope

This is a results-reproduction release. Complete original API requests and literature-containing dialogues are omitted, as are credentials and account records. The archived generation/docking scripts document historical implementation and retain their historical path assumptions; they are not portable turnkey experiment commands. The portable entry points are the two commands above. The immutable weights of the external served language model and the complete original GIN training lineage are not available. `models/encoder_source.json` contains the two source functions archived with Study B, rather than a complete training package.

The manuscript and editable schematic artwork are not published by this release. Partial release boundaries are described in [RELEASE_SCOPE.md](docs/RELEASE_SCOPE.md). Public access does not by itself establish journal compliance or independent replication. See [TERMS.md](TERMS.md) for reuse terms and third-party attribution.
