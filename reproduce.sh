#!/usr/bin/env bash
set -euo pipefail

python -m pip install -r requirements.lock
python -m pip install -e .

if [[ "${1:-}" == "--full-gpu" ]]; then
  python -m cattle_id.hardening_matrix --matrix configs/experiment_matrix_hardening_v2.yaml
  python -m cattle_id.hardening_postprocess --holstein-metadata artifacts/metadata/holstein2025_open_set.csv
  python -u tools/finalize_hardening_colab.py
  exit 0
fi

evidence_zip="${1:-${HARDENING_EVIDENCE_ZIP:-}}"
if [[ -z "$evidence_zip" ]]; then
  echo "Usage: ./reproduce.sh /path/to/hardening_v2_evidence.zip" >&2
  echo "For the complete GPU matrix, use ./reproduce.sh --full-gpu" >&2
  exit 2
fi

python -u tools/reproduce_release.py --evidence "$evidence_zip"
python -m pytest -q -p no:cacheprovider
