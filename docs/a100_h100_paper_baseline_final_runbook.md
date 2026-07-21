# Final-Scope A100/H100 Runbook

Protocol-aligned paper replication with one shared seed and shared test metadata.
The historical 240-job matrix is not part of this final acceptance contract.

## Matrix Summary

Base config: configs/cattlessfr_colab.yaml
Seeds: 2026
Models: vgg16, resnet50, mobilenetv2, densenet121, efficientnetb0
Protocols: paper_random
Evaluations: none
Train jobs: 5
Evaluation jobs: 0
Total jobs: 5
Colab Pro+ notebook: notebooks/colab_train.ipynb

## Completion Contract

1. Reuse completed run directories only when all required artifacts validate.
2. Keep raw datasets in the local Colab runtime and evidence under Google Drive.
3. Generate metrics and predictions before marking a training job complete.
4. Regenerate scoped CSV/TEX tables from artifacts; do not type numerical results manually.
5. Package final metrics, predictions, manifests and SHA-256 hashes for local thesis integration.

## Commands

```powershell
python -m cattle_id.run_matrix --matrix configs/experiment_matrix_paper_baseline_final.yaml
python tools/evidence_summary.py --runs artifacts/runs --scope configs/final_evidence_scope.yaml
python tools/final_preflight.py -s
```

## Claim Boundary

The final evidence supports protocol-aligned replication, bounded transform robustness and public unseen-identity transfer. It does not establish field generalisation on new captures of the original CattleSSFR animals.
