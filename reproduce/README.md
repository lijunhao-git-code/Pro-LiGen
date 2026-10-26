# Portable reproduction entry points

Run the commands in the repository root README. `offline_analysis.py` recomputes the A/B/C summaries, target-paired tests, candidate replay and docking from row-level observations; its scientific functions are identical to the manuscript's retained analysis. Only distribution metadata was updated for this release.

`render_figures.py` calls the original retained plotting functions using `figure_inputs_frozen.json`. It renders only quantitative result figures; schematic drawings are not regenerated. `figure_text_overrides.py` applies the approved label changes and checks that numeric coordinates are unchanged. PDF timestamps and font rendering can vary by environment.

`expected_frozen_results.json` supplies verification targets, not the source observations used to compute the statistics. Core and optional dependency versions are pinned separately. The optional chemistry check verifies 1,023 retained parent/reference records; this is a record count, not a count of distinct molecules.
