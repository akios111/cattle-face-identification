# A100/H100 Open-Set Experiment Runbook

This runbook executes the no-shortcut public unseen-identity route after same-identity external acquisition was confirmed unavailable.
It does not describe Holstein2025 as a same-identity external acquisition holdout for CattleSSFR.

## Matrix Summary

Source config: configs/cattlessfr_colab_proplus.yaml
Target config: configs/holstein2025_colab_proplus.yaml
Seeds: 1, 2, 3, 4, 5
Models: efficientnetv2b3, convnexttiny, efficientnetb0
Source protocols: paper_random, transform_holdout, ablation_geometric_only, ablation_no_cutout, ablation_cutout_only, ablation_224, ablation_frozen
Evaluations: holstein2025_zero_shot_reid, holstein2025_in_domain_reid
Source train jobs: 105
Target train jobs: 15
Train jobs: 120
Evaluation jobs: 120
Total jobs: 240
Holstein2025 metadata: artifacts/metadata/holstein2025_open_set.csv
Colab Pro+ notebook: notebooks/colab_train.ipynb

## Required Gates

1. Validate the pinned Holstein2025 snapshot and SHA-256 manifest.
2. Run five seed shards with deterministic run IDs and skip-completed semantics.
3. Evaluate every CattleSSFR run zero-shot on unseen Holstein2025 identities.
4. Train on the 82 Holstein2025 development identities and evaluate the 20 unseen identities.
5. Report CMC@1, CMC@5, mAP and animal-level bootstrap intervals.

## Commands

```powershell
python tools/holstein2025_readiness.py
python -m cattle_id.run_matrix --matrix configs/experiment_matrix_open_set.yaml --out artifacts/matrix/experiment_matrix_open_set_jobs.json
python tools/evidence_summary.py
python tools/final_preflight.py -s
```

## Claim Boundary

Holstein2025 supports independent public unseen-identity evidence. It does not retroactively create new captures of the 311 CattleSSFR animals, and no claim may describe it as same-identity field validation.
