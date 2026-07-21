# Dataset access

The archive contains dataset metadata, relative split manifests, SHA-256 manifests,
labels, and published reference metrics. It intentionally contains no raw CattleSSFR or
Holstein2025 image files.

Acquire the pinned datasets from their original repositories with:

```powershell
python scripts/acquire_datasets.py --help
```

Pinned revisions:

- CattleSSFR: `099d749e9a766ff0c9b9fbc49112c6b77b29542e`
- Holstein2025: `b905600ca4153e8435c1a2c33306a2783de6fbdf`

This policy prevents the research package from asserting redistribution rights that the
upstream repositories do not explicitly grant.
