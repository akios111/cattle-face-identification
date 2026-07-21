# hardening_v2 Colab Pro+ Runbook

This is the only GPU route for the final hardening evidence. It executes 29 trainings and 16 Holstein2025 evaluations sequentially and resumes from validated Drive artifacts. No shard or job index must be edited.

## 1. Build the upload bundle locally

```powershell
python tools/build_colab_hardening_notebook.py
python tools/build_colab_bundle.py
```

Upload these two files to `MyDrive`:

- `artifacts/deployment/cattle_face_identification_hardening_v2_bundle.zip`
- `artifacts/deployment/cattle_face_identification_hardening_v2_bundle.sha256.txt`

The notebook checks the sidecar hash before applying the bundle.

## 2. Run Colab

1. Open `notebooks/colab_train.ipynb` in Colab Pro+.
2. Select an A100 or H100 runtime.
3. Choose **Runtime > Run all**.
4. Leave all matrix controls unchanged.

The notebook writes each completed job immediately below:

```text
/content/drive/MyDrive/cattle_face_identification/artifacts/
```

Reconnecting or restarting the runtime is safe. A job is skipped only when its checkpoint, predictions, metrics, manifest, hash, and completion marker all validate.

If all jobs are complete but the runtime restarted before final packaging, mount Drive, change to the persistent project directory, and run only:

```bash
python -u tools/finalize_hardening_colab.py
```

The finalizer requires the `29/29` and `16/16` manifests before it starts. It does not contain a training or evaluation command. Existing complete summary tables are reused; the pinned CattleSSFR and Holstein2025 snapshots are reacquired under `/content/cattle_runtime` only when an image-based figure needs them. The 20 montage variants and the selected Holstein error images are verified against the recorded SHA-256 values before rendering. Raw datasets remain runtime-only and are excluded from the evidence ZIP.

## 3. Retrieve the final evidence

Wait until the notebook reports both contracts complete and verifies this file:

```text
/content/drive/MyDrive/cattle_face_identification/artifacts/evidence/hardening_v2_evidence.zip
```

Download that ZIP. Raw dataset images are intentionally excluded.

## 4. Import and validate locally

```powershell
python tools/import_hardening_evidence.py "C:\path\to\hardening_v2_evidence.zip"
python tools/hardening_v2_contract.py --strict
python tools/section_contracts.py
```

The strict contract must report:

```text
hardening_v2_complete=True trainings=29/29 holstein=16/16 bundle=True
```

Only after this point are the generated numerical fragments eligible for the thesis, the new GATE 3 review, and final PDF/DOCX production.
