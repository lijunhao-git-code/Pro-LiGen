# Deployed GIN material

`best_gin_rc.pth` is the frozen checkpoint used by Study B. `encoder_source.json` preserves the two source functions archived with that experiment and the original run.py hash. `../data/retrieval/library_embeddings.npy` preserves the deployed library embeddings in the row order of `library.csv`.

The complete executable feature extraction, graph builder and GIN encoder are in `../proligen/gin.py`. All four definitions are extracted unchanged from the original server `run.py`, preserved in `../archive/original_project/run.py`. Function-level AST identity is recorded in `../provenance/gin_source_restoration.json`.

Run `python reproduce/retrieve_references.py --out gin_run_cached` from the repository root. Add `--recompute-library` to encode the library from SMILES as well. See `../docs/GIN_REPRODUCTION.md` for instructions and the verification reports under `../verification/` for both modes.

The retained training triplets, training curve and original log are under `../data/training/`; the trainer is `../archive/train_gin.py`. These files are independent of the frozen-inference command and are not modified by it.
