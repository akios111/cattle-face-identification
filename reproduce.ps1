param(
    [string]$Evidence,
    [switch]$FullGpu
)

$ErrorActionPreference = "Stop"

python -m pip install -r requirements.lock
python -m pip install -e .

if ($FullGpu) {
    python -m cattle_id.hardening_matrix --matrix configs/experiment_matrix_hardening_v2.yaml
    python -m cattle_id.hardening_postprocess --holstein-metadata artifacts/metadata/holstein2025_open_set.csv
    python -u tools/finalize_hardening_colab.py
    exit 0
}

if (-not $Evidence) {
    $Evidence = $env:HARDENING_EVIDENCE_ZIP
}
if (-not $Evidence) {
    throw "Usage: .\\reproduce.ps1 -Evidence C:\\path\\to\\hardening_v2_evidence.zip"
}

python -u tools/reproduce_release.py --evidence $Evidence
python -m pytest -q -p no:cacheprovider
