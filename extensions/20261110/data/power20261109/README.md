# Study-B ranking-effect sensitivity

These records summarize minimum detectable differences in QED yield U under a paired-t model. They are a variance-based sensitivity analysis, not an exact power calculation for the original sign-flip/Holm tests and not evidence of equivalence.

Inputs are the frozen ring and LogP/TPSA task means in data/updated20261102. Five runs are already averaged within each of 25 paired tasks. The observed standard deviation of those paired task differences determines each MDE. Family sizes are 4 for ring and 8 for the joint LogP/TPSA family, retaining the original analysis identities.

- MDE9.csv: nine M-R, G-R and G-M contrasts; single-test alpha=.05 and conservative alpha=.05/family_size.
- paired_task_differences225.csv: the underlying 25 differences for each contrast.
- method.json: assumptions, numerical calibration, source hashes and documentation.
- verification.json: verification summary.

Unpack the supporting ZIP, enter Pro_LiGen_JCIM, then run `python reproduce/power_sensitivity.py`. Python >=3.10 and NumPy are required. The command verifies the archived outputs offline; `--write` regenerates them. No API, model training or docking is invoked. The implementation evaluates noncentral-t power using its normal/chi-square mixture, avoiding a SciPy installation. Quadrature orders 128 and 256 are cross-checked.
