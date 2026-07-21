# External Acquisition Collection Protocol

This protocol defines the private real-image acquisition required before any field-generalization claim is upgraded. No shortcut: public datasets, repeated original CattleSSFR frames, augmented images, synthetic images, or generated examples cannot populate `data/external_acquisition/manifest.csv`.

## Acceptance Target

- Collect at least 100 animals.
- Capture at least 3 images per animal.
- Use a different day, lighting, pose, or operator context from the original CattleSSFR material whenever possible.
- Keep every row in split `test`; no external_acquisition_holdout image may enter train or validation.
- Preserve a SHA-256 manifest for images, annotations, trained runs, and generated evidence tables.

## Manifest Rules

Use `data/external_acquisition/manifest.template.csv` as the import contract. The final file is `data/external_acquisition/manifest.csv`, which remains private/local until the real acquisition exists. Each `class_id` must map to the CattleSSFR identity used by the trained classifiers. Each `acquisition_id` must be unique and must not overlap with any training or validation metadata.

## Capture Rules

Store images under `data/external_acquisition/images/<animal_id>/`. Prefer full-face and three-quarter views, but keep difficult field cases instead of cleaning the set into an easy benchmark. Record the camera, lighting condition, pose, operator, and notes immediately after capture.

## Masking Annotation Rules

Use `data/external_acquisition/masking_annotations.template.csv` for shortcut checks. Mark absent ear tags, paint marks, face boxes, or background regions as `not_present` and leave the coordinate columns empty. Do not encode absence as missing metadata.

## Evidence Handoff

After acquisition, run external validation readiness, SHA-256 manifest generation, the A100/H100 matrix, evidence summary generation, and final preflight. The thesis may only cite generated CSV/TEX tables from completed runs.
