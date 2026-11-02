# Property-specific GIN inputs and verification

Ring selection retains `models/best_gin_rc.pth` and the original public retrieval entry point. LogP and TPSA adaptations use a shared initializer with SHA256 `ad65679dc6b9a9b63fe2f7efd97fb1fd115b2b92c14303d6c492807e725266fe`.

| Property | Train / validation triplets | Epochs / selected epoch | Actual checkpoint SHA256 |
|---|---|---|---|
| LogP |17,334 /3,650|34 /24|`76e4dab6557a77f08a365b5d5bb671731dab106469868b9298aa587879efdf41`|
| TPSA |16,187 /3,426|29 /19|`0fe2bf857b14ecfcb6731244b1dcf2bd4dedd93ab841190b8ae0ea5bb3e96076`|

`gin/code/gin_model_original.py` is the actual deployed feature/encoder/training implementation. Inputs have84 atom features; the shared architecture uses five layers with 256-dimensional hidden/output states and mean pooling. Retrieval L2-normalizes embeddings. Training preserves the original dot-product contrastive objective and validation-loss checkpoint selection.

The property library is the 24,693 valid single-component carbon-containing structures from the original26,108-entry library, with original split and source identifiers retained. Exact task seeds are excluded from adaptation inputs. The four triplet tables, both epoch curves, selected checkpoints, shared initializer, labels and exact input hashes are provided. `graphs.pt` and dense float32 Morgan matrices are reconstructible caches and are not duplicated; the explicit training wrapper rebuilds the graph cache.

Both query and library graph encoding can be checked on CPU. Cached-library validation reselected all 2,250 supplied M/G/R reference records exactly in the validated Python 3.10, torch 2.7.1, RDKit 2026.3.4 environment. Full-library re-encoding additionally checks numerical agreement with the saved float32 embeddings; near-tie rankings are recorded separately if hardware/reduction rounding changes their order. The originally frozen examples remain unchanged in the public records.

See `verification/current_*.json` for executed checks. None of the validation commands updates the pretrained weights or calls a generation API.
