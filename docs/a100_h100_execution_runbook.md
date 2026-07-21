# A100/H100 Experiment Runbook

This runbook expands the approved 95-100/100 hardening matrix for GPU execution.
No 100/100 field-generalization claim is allowed until data/external_acquisition/manifest.csv contains real independent acquisitions and the external validation readiness gate passes.

## Matrix Summary

Base config: configs/cattlessfr_colab_proplus.yaml
Seeds: 1, 2, 3, 4, 5
Models: efficientnetv2b3, convnexttiny, efficientnetb0
Protocols: paper_random, transform_holdout, ablation_geometric_only, ablation_no_cutout, ablation_cutout_only, ablation_224, ablation_frozen
Evaluations: external_acquisition_holdout, mask_ear_tag, mask_paint, mask_background
Train jobs: 105
Evaluation jobs: 60
Total jobs: 165
External metadata: artifacts/metadata/external_acquisition_holdout.csv
Masking annotations: data/external_acquisition/masking_annotations.csv
Colab Pro+ notebook: notebooks/colab_train.ipynb

## Required Gates

1. Build and validate the private external acquisition manifest.
2. Generate SHA-256 hashes for external images and annotations.
3. Run the full seed/model/protocol matrix on A100/H100.
4. Evaluate external_acquisition_holdout and masking shortcut variants.
5. Regenerate thesis CSV/TEX evidence before changing claims.

## Commands

```powershell
python tools/external_validation_readiness.py --config configs/external_acquisition.yaml --out thesis/gates/external-validation-readiness.md
python -m cattle_id.run_matrix --matrix configs/experiment_matrix_100.yaml --out artifacts/matrix/experiment_matrix_100_jobs.json
python tools/evidence_summary.py
python tools/final_preflight.py -s
```

## Claim Boundary

The Cattely public benchmark remains supportive external evidence, not a substitute for the real external_acquisition_holdout. The final thesis claim must only use rows backed by completed jobs and generated evidence tables.
