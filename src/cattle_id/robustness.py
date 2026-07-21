from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps

from .hashing import sha256_file


REGION_VARIANTS = (
    "grayscale",
    "central_region_only",
    "peripheral_region_only",
    "random_region_only",
    "random_area_mask",
)

SEVERITY_LEVELS: dict[str, tuple[float, ...]] = {
    "rotation": (0.0, 5.0, 10.0, 15.0, 20.0),
    "gaussian_noise": (0.0, 6.0, 12.0, 18.0, 24.0),
    "blur": (0.0, 0.7, 1.4, 2.1, 2.8),
    "cutout_center": (0.0, 0.125, 0.25, 0.375, 0.5),
}


def _rng(seed: int, sample_id: str, variant: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}\x1f{sample_id}\x1f{variant}".encode("utf-8")).digest()
    return np.random.default_rng(seed + int.from_bytes(digest[:8], "big"))


def _region_box(
    image: Image.Image,
    *,
    fraction: float = 0.70,
    random_position: bool = False,
    rng: np.random.Generator | None = None,
) -> tuple[int, int, int, int]:
    width, height = image.size
    region_width = max(1, int(round(width * fraction)))
    region_height = max(1, int(round(height * fraction)))
    if random_position:
        if rng is None:
            raise ValueError("random region placement requires an RNG")
        left = int(rng.integers(0, max(1, width - region_width + 1)))
        top = int(rng.integers(0, max(1, height - region_height + 1)))
    else:
        left = (width - region_width) // 2
        top = (height - region_height) // 2
    return left, top, left + region_width, top + region_height


def apply_region_variant(
    image: Image.Image,
    variant: str,
    *,
    sample_id: str,
    seed: int = 1,
) -> Image.Image:
    if variant not in REGION_VARIANTS:
        raise ValueError(f"Unknown image-region variant: {variant}")
    image = image.convert("RGB")
    if variant == "grayscale":
        return ImageOps.grayscale(image).convert("RGB")

    random_position = variant in {"random_region_only", "random_area_mask"}
    generator = _rng(seed, sample_id, variant)
    box = _region_box(
        image,
        random_position=random_position,
        rng=generator,
    )
    neutral = (127, 127, 127)
    if variant in {"central_region_only", "random_region_only"}:
        output = Image.new("RGB", image.size, neutral)
        output.paste(image.crop(box), box)
        return output
    output = image.copy()
    patch = Image.new("RGB", (box[2] - box[0], box[3] - box[1]), neutral)
    output.paste(patch, box)
    return output


def apply_severity_transform(
    image: Image.Image,
    family: str,
    value: float,
    *,
    sample_id: str,
    direction: int = 1,
    seed: int = 1,
) -> Image.Image:
    if family not in SEVERITY_LEVELS:
        raise ValueError(f"Unknown severity family: {family}")
    image = image.convert("RGB")
    if family == "rotation":
        return image.rotate(
            float(value) * int(direction),
            resample=Image.Resampling.BILINEAR,
            fillcolor=(0, 0, 0),
        )
    if family == "gaussian_noise":
        pixels = np.asarray(image, dtype=np.float32)
        noise = _rng(seed, sample_id, f"{family}:{value}").normal(0.0, value, pixels.shape)
        return Image.fromarray(np.uint8(np.clip(pixels + noise, 0, 255)), mode="RGB")
    if family == "blur":
        return image.filter(ImageFilter.GaussianBlur(radius=float(value)))
    if value <= 0:
        return image.copy()
    width, height = image.size
    cut_width = max(1, int(round(width * value)))
    cut_height = max(1, int(round(height * value)))
    left = (width - cut_width) // 2
    top = (height - cut_height) // 2
    output = image.copy()
    output.paste(Image.new("RGB", (cut_width, cut_height), (0, 0, 0)), (left, top))
    return output


def _target_path(output_dir: Path, sample_id: str, suffix: str) -> Path:
    digest = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:16]
    return output_dir / f"{digest}__{suffix}.png"


def materialize_region_audit(
    metadata: pd.DataFrame,
    output_dir: str | Path,
    *,
    variants: tuple[str, ...] = REGION_VARIANTS,
    seed: int = 1,
) -> pd.DataFrame:
    required = {"sample_id", "image_path", "class_id", "split"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"region audit metadata missing columns: {', '.join(missing)}")
    source = metadata[metadata["split"] == "test"].sort_values("sample_id")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        with Image.open(str(row["image_path"])) as opened:
            image = opened.convert("RGB")
        for variant in variants:
            transformed = apply_region_variant(
                image,
                variant,
                sample_id=str(row["sample_id"]),
                seed=seed,
            )
            target = _target_path(output_dir, str(row["sample_id"]), variant)
            transformed.save(target)
            updated = dict(row)
            updated["original_sample_id"] = str(row["sample_id"])
            updated["sample_id"] = f"{row['sample_id']}::region::{variant}"
            updated["image_path"] = str(target.resolve())
            updated["image_sha256"] = sha256_file(target)
            updated["augmentation_id"] = variant
            updated["augmentation_family"] = "image_region_audit"
            updated["region_variant"] = variant
            updated["split"] = "test"
            rows.append(updated)
    return pd.DataFrame(rows)


def materialize_severity_sweep(
    metadata: pd.DataFrame,
    output_dir: str | Path,
    *,
    seed: int = 1,
) -> pd.DataFrame:
    required = {"sample_id", "image_path", "class_id", "augmentation_id"}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"severity metadata missing columns: {', '.join(missing)}")
    source = metadata[metadata["augmentation_id"] == "original"].sort_values("sample_id")
    if source.empty:
        raise ValueError("severity sweep requires one original image per identity")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for row in source.to_dict(orient="records"):
        with Image.open(str(row["image_path"])) as opened:
            image = opened.convert("RGB")
        for family, levels in SEVERITY_LEVELS.items():
            for value in levels:
                directions = (-1, 1) if family == "rotation" and value > 0 else (1,)
                for direction in directions:
                    suffix = f"{family}_{'neg' if direction < 0 else 'pos'}_{value:g}"
                    transformed = apply_severity_transform(
                        image,
                        family,
                        value,
                        sample_id=str(row["sample_id"]),
                        direction=direction,
                        seed=seed,
                    )
                    target = _target_path(output_dir, str(row["sample_id"]), suffix)
                    transformed.save(target)
                    updated = dict(row)
                    updated["original_sample_id"] = str(row["sample_id"])
                    updated["sample_id"] = f"{row['sample_id']}::severity::{suffix}"
                    updated["image_path"] = str(target.resolve())
                    updated["image_sha256"] = sha256_file(target)
                    updated["augmentation_id"] = suffix
                    updated["augmentation_family"] = "severity_sweep"
                    updated["severity_family"] = family
                    updated["severity_value"] = float(value)
                    updated["severity_direction"] = int(direction)
                    updated["split"] = "test"
                    rows.append(updated)
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize deterministic robustness evaluations.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("region", "severity"), required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--metadata-out", required=True)
    args = parser.parse_args(argv)
    metadata = pd.read_csv(args.metadata)
    if args.mode == "region":
        output = materialize_region_audit(metadata, args.out, seed=args.seed)
    else:
        output = materialize_severity_sweep(metadata, args.out, seed=args.seed)
    metadata_out = Path(args.metadata_out)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(metadata_out, index=False)
    print(metadata_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
