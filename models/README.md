# Deployed GIN material

`best_gin_rc.pth` is the frozen checkpoint used by Study B. It is not a newly trained model. `encoder_source.json` preserves the two source functions archived with that experiment and the original run.py hash; graph-feature helper definitions are not present in that snapshot. `../data/retrieval/library_embeddings.npy` preserves the deployed library embeddings in the row order of `library.csv`. This first release does not present the incomplete snapshot as a complete training/inference package. See the scope document.
