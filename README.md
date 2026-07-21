# Deep Learning for Cattle Face Identification

Reproducible evaluation of CattleSSFR as a single-source-image benchmark, with frozen byte manifests, independently controlled training/split/augmentation seeds, shortcut baselines, controlled ablations, robustness tests, and zero-shot Holstein2025 retrieval controls.

The repository distinguishes two evidence generations:

- `legacy_replication`: the historical paper-aligned experiments and already archived thesis evidence.
- `hardening_v2`: the corrected augmentation RNG, byte-frozen test sets, 29 controlled trainings, and 16 Holstein2025 evaluations.

No result from one generation is silently relabelled as belonging to the other.

## Reproduce the released evidence

From a clean clone, download the immutable evidence asset from the
[`v1.0.0` release](https://github.com/akios111/cattle-face-identification/releases/tag/v1.0.0)
and run:

```bash
gh release download v1.0.0 --repo akios111/cattle-face-identification --pattern hardening_v2_evidence.zip --dir release-assets
./reproduce.sh release-assets/hardening_v2_evidence.zip
```

On Windows PowerShell, use:

```powershell
gh release download v1.0.0 --repo akios111/cattle-face-identification --pattern hardening_v2_evidence.zip --dir release-assets
.\reproduce.ps1 -Evidence release-assets\hardening_v2_evidence.zip
```

This CPU route verifies every bundle hash, imports only whitelisted paths, regenerates the 17 central CSV tables in an isolated directory, compares their schema and categorical values exactly and numerical values within an absolute `1e-12` tolerance, verifies four auxiliary statistical artifacts, verifies the frozen--ImageNet state audit, checks the `29/29` and `16/16` contract, and runs the test suite. Raw images, model training, thesis-generation scripts, and internal audit scripts are not required for this evidence-level reproduction.

## Verify the release assets

The public release has three large assets: the primary paper-random seed-1 checkpoint, the complete hardening evidence ZIP, and the visual addendum. Their release names, byte sizes, SHA-256 values, protocol fields, and source paths are generated rather than typed manually:

```powershell
python tools/release_asset_manifest.py
python tools/release_asset_manifest.py --verify-only
```

The tracked outputs are `release/v1.0.0/ASSET_MANIFEST.json` and `release/v1.0.0/SHA256SUMS`. The checkpoint and ZIP files remain outside Git and are uploaded as GitHub/Zenodo release assets. Raw CattleSSFR and Holstein2025 images are not included.

## Reproduce the hardening matrix

The GPU route is `notebooks/colab_train.ipynb`. On Colab Pro+ with an A100 or H100, choose **Runtime > Run all**. The notebook performs every job sequentially and resumes from validated Google Drive artifacts without a shard index.

The command-line equivalent is:

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -e .
python -m cattle_id.hardening_matrix --matrix configs/experiment_matrix_hardening_v2.yaml
python -m cattle_id.hardening_postprocess --holstein-metadata artifacts/metadata/holstein2025_open_set.csv
python tools/hardening_evidence_summary.py
python tools/ensure_hardening_figure_inputs.py
python tools/render_hardening_figures.py
python tools/build_hardening_evidence.py --out artifacts/evidence/hardening_v2_evidence.zip
python tools/verify_hardening_evidence.py --zip artifacts/evidence/hardening_v2_evidence.zip
```

The complete GPU route is `./reproduce.sh --full-gpu` or `.\reproduce.ps1 -FullGpu`. Completed training and evaluation jobs are skipped only when their required manifests, predictions, metrics, checkpoint hashes, and completion markers validate.

After a Colab runtime restart, run `python -u tools/finalize_hardening_colab.py` from the persistent Drive project. It validates the `29/29` and `16/16` completion manifests, skips already complete summary tables, restores only the pinned and SHA-256-verified runtime images required by the figures, and performs no training or evaluation.

After the Colab run, download only `artifacts/evidence/hardening_v2_evidence.zip` from the persistent Drive project and import it locally:

```powershell
python tools/import_hardening_evidence.py C:\path\to\hardening_v2_evidence.zip
python tools/hardening_v2_contract.py --strict
```

The importer verifies the bundle manifest and every SHA-256 before writing generated tables, figures, predictions, embeddings, and thesis fragments. It rejects path traversal, raw-image roots, undeclared files, and evidence produced from configs that differ from the local immutable hardening configs. See `docs/hardening_v2_colab_runbook.md` for the end-to-end handoff.

## Data policy

Raw CattleSSFR and Holstein2025 images are not distributed here because their upstream repositories do not declare an explicit redistribution license. `scripts/acquire_datasets.py` obtains the pinned snapshots from the original repositories:

- CattleSSFR: `099d749e9a766ff0c9b9fbc49112c6b77b29542e`
- Holstein2025: `b905600ca4153e8435c1a2c33306a2783de6fbdf`

The release contains acquisition code, relative manifests, checksums, predictions, embeddings, generated tables, figures, and permitted model artifacts, but not the raw datasets.

`DATA_LICENSE.md` separates the MIT-licensed code, CC BY 4.0 project-generated evidence, and upstream image collections for which this project grants no redistribution rights.

## Scientific claim boundary

Near-perfect CattleSSFR accuracy means classification of held-out transformations derived from the same 311 source photographs. It does not establish recognition from new real captures of those animals. Holstein2025 evaluates transfer to unseen identities in another dataset; it is not same-animal field validation.

## Tests

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The thesis source is under `thesis/`. Numerical tables used by the hardening revision are generated from evidence artifacts and are never typed manually.
