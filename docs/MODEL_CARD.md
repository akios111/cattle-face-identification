# EfficientNetV2B3 hardening_v2 model card

## Intended use

The checkpoints support research on transformed-source classification in CattleSSFR and embedding-based evaluation on unseen Holstein2025 identities. They are not validated for autonomous farm decisions, animal welfare interventions, surveillance, or recognition from new captures of the original CattleSSFR animals.

## Training data

CattleSSFR contains one source photograph for each of 311 identities. The hardening pipeline creates nineteen deterministic transformations plus the source image. Every materialized image is tied to a pinned upstream commit and a SHA-256 manifest. Raw images are acquired from the upstream repository and are not redistributed in this release.

## Evaluation

The release reports fixed-split training-seed variability, repeated split sensitivity, three paired ablations, simple source-image baselines, geometric and severity controls, Grad-CAM faithfulness, and zero-shot Holstein2025 retrieval against an ImageNet-only control.

## Limitations

High closed-set accuracy can reflect matching of transformed versions of one source photograph. Holstein2025 contains different identities and acquisition conditions, so its results measure cross-dataset representation transfer rather than same-animal generalization.
