from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from cattle_id.hashing import write_sha256_manifest, write_sha256_manifest_from_csv_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a SHA-256 CSV manifest for files or directories.")
    parser.add_argument("paths", nargs="*", help="Files or directories to hash recursively.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--from-csv-manifest", type=Path)
    parser.add_argument("--image-path-column", default="image_path")
    args = parser.parse_args(argv)

    if args.from_csv_manifest is not None:
        output = write_sha256_manifest_from_csv_manifest(
            args.from_csv_manifest,
            args.out,
            image_path_column=args.image_path_column,
            root=args.root,
        )
    else:
        if not args.paths:
            parser.error("paths are required unless --from-csv-manifest is provided")
        output = write_sha256_manifest(args.paths, args.out, root=args.root)
    print(f"sha256_manifest={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
