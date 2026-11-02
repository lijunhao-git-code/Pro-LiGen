# Runnable stages and preserved records

| Entry point | Executed stage | API / training / docking |
|---|---|---|
| `reproduce_current_paper.py --out NEW` | Initial and extended numerical summaries, inference and candidate-policy replay | None |
| Same command with `--figures` | Numerical stages plus actual quantitative figure regeneration | None |
| `verify_extended_selection.py --out NEW` | Exact final identity and full candidate-dictionary replay, all 1,500 runs | None |
| `retrieve_property_references.py --out NEW` | Morgan/R sampling and frozen-GIN query inference against saved library embeddings | Inference only |
| Same command with `--recompute-library` | Also graph-encodes all 24,693 molecules using both property encoders | Inference only |
| `train_property_adaptations.py --out NEW` | Triplet, split, property labels and one forward-loss validation | No optimizer step |
| Same command with `--run-training` | Rebuild graph cache and execute original property trainer in a new directory | Training, no model API |

Original generation and scoring implementations are preserved under the extension's `gin/code/`, `archive/ring_code/` and scoring directories. Their source assumptions and model-service metadata remain archived. Exact historical API responses are scientific observations, not a promise that a later stochastic API call returns the same molecule. No default reproduction command submits paid calls. Prompt templates are in the original generator/preparation code; full third-party literature-containing dialogues are outside the distribution.

The training check validates archived inputs and implementation; a new full training run is a separate explicit action. Model/library/record hashes are in the collected-source manifest and top-level release manifest. Inference reference checks report exact order and, separately, any float32 near-ties without changing the original frozen input examples or introducing a new selection rule.
