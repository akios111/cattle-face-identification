from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image, ImageFilter


REGION_BY_VARIANT = {
    "mask_ear_tag": "ear_tag",
    "mask_paint": "paint_mark",
    "mask_background": "background",
}


def _is_missing(value: object) -> bool:
    return value is None or (isinstance(value, float) and pd.isna(value)) or str(value).strip() == ""


def _region(row: pd.Series, prefix: str) -> tuple[int, int, int, int] | None:
    keys = [f"{prefix}_{axis}" for axis in ("x1", "y1", "x2", "y2")]
    if any(_is_missing(row.get(key)) for key in keys):
        return None
    values = tuple(int(float(row[key])) for key in keys)
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid {prefix} region: {values}")
    return values


def _masked_image(image: Image.Image, region: tuple[int, int, int, int], variant: str) -> Image.Image:
    output = image.convert("RGB").copy()
    if variant == "mask_background":
        blurred = output.filter(ImageFilter.GaussianBlur(radius=8))
        output.paste(blurred.crop(region), region)
        return output
    patch = Image.new("RGB", (region[2] - region[0], region[3] - region[1]), (0, 0, 0))
    output.paste(patch, (region[0], region[1]))
    return output


def apply_masking_annotations(
    metadata: pd.DataFrame,
    annotations: pd.DataFrame,
    output_dir: str | Path,
    *,
    variant: str,
) -> pd.DataFrame:
    if variant not in REGION_BY_VARIANT:
        raise ValueError(f"Unknown masking variant: {variant}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    region_prefix = REGION_BY_VARIANT[variant]
    annotations_by_path = {
        str(Path(row["image_path"]).resolve()): row
        for _, row in annotations.iterrows()
        if "image_path" in row
    }

    rows: list[dict[str, object]] = []
    for _, row in metadata.iterrows():
        source_path = Path(str(row["image_path"])).resolve()
        annotation = annotations_by_path.get(str(source_path))
        if annotation is None:
            raise ValueError(f"missing masking annotation for image: {source_path}")
        image = Image.open(source_path).convert("RGB")
        region = _region(annotation, region_prefix)
        mask_status = "applied"
        if region is None:
            output_image = image.copy()
            mask_status = "not_present"
        else:
            output_image = _masked_image(image, region, variant)
        target = output_dir / f"{source_path.stem}__{variant}{source_path.suffix.lower()}"
        output_image.save(target)
        updated = row.to_dict()
        updated["image_path"] = str(target.resolve())
        updated["augmentation_id"] = variant
        updated["augmentation_family"] = "masking"
        updated["mask_status"] = mask_status
        rows.append(updated)
    return pd.DataFrame(rows)
