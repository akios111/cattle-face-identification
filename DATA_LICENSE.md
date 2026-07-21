# Data and Evidence Licensing

## Project-created material

The source code and original project documentation in this repository are released
under the MIT License in `LICENSE`.

Project-generated numerical evidence, including predictions, embeddings, metrics,
tables, manifests, and figures, is released under the Creative Commons Attribution
4.0 International license (CC BY 4.0):
https://creativecommons.org/licenses/by/4.0/

The released EfficientNetV2B3 checkpoint is a project-generated model artifact and
is distributed under the MIT License. Its use does not grant any rights in the source
dataset images.

## Upstream datasets

Raw CattleSSFR and Holstein2025 images are not redistributed by this project. At the
pinned revisions listed below, neither upstream repository contained a `LICENSE`,
`LICENCE`, or `COPYING` file, and GitHub reported no detected SPDX license:

| Dataset | Upstream repository | Pinned commit |
| --- | --- | --- |
| CattleSSFR | https://github.com/MachineLearningVisionRG/CattleSSFR | `099d749e9a766ff0c9b9fbc49112c6b77b29542e` |
| Holstein2025 | https://github.com/JZM-shuimu/Cattle-ID | `b905600ca4153e8435c1a2c33306a2783de6fbdf` |

This status was checked on 2026-07-21. Absence of a repository license is not a grant
of permission. Users must obtain the datasets from their original repositories and
comply with any terms supplied by the copyright holders. The acquisition helper
`scripts/acquire_datasets.py` checks out only the pinned revisions; it does not confer
redistribution or commercial-use rights.

## Included metadata

The public release contains relative paths, labels, split assignments, source commit
identifiers, and SHA-256 values needed to verify experimental provenance. It contains
no raw dataset image bytes. Hashes and paths identify the evaluated inputs but do not
replace the upstream datasets.
