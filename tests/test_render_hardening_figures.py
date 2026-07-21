from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image


sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from render_hardening_figures import (
    HOLSTEIN_SUFFIX,
    _load_gradcam_heatmap,
    render_error_gallery,
    render_pipeline,
)


def test_hardening_pipeline_figure_is_nonblank_and_wide(tmp_path: Path):
    output = render_pipeline(tmp_path / "pipeline.png")

    with Image.open(output) as image:
        assert image.width > image.height * 3
        colors = image.convert("RGB").getcolors(maxcolors=image.width * image.height)
        assert colors is not None
        assert len(colors) > 10


def test_gradcam_heatmap_is_scaled_to_display_range(tmp_path: Path):
    source = tmp_path / "heatmap.png"
    pixels = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    Image.fromarray(pixels, mode="L").save(source)

    heatmap = _load_gradcam_heatmap(source)

    assert heatmap.dtype == np.float32
    assert float(heatmap.min()) == 0.0
    assert float(heatmap.max()) == 1.0
    assert np.isclose(float(heatmap[0, 1]), 64 / 255)


def test_holstein_error_gallery_resolves_ephemeral_path_from_relative_path(tmp_path: Path):
    runs = tmp_path / "runs"
    run_id = "fine-tuned-run"
    run_dir = runs / run_id
    run_dir.mkdir(parents=True)
    holstein_root = tmp_path / "Holstein2025"
    image_path = holstein_root / "datasets_v2" / "query1" / "animal-a" / "probe.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), (30, 70, 110)).save(image_path)
    pd.DataFrame(
        [
            {
                "image_path": "/content/cattle_runtime/raw/Holstein2025/missing.jpg",
                "relative_path": image_path.relative_to(holstein_root).as_posix(),
                "animal_id": "animal-a",
                "predicted_animal_id": "animal-b",
                "correct_rank_1": False,
            }
        ]
    ).to_csv(run_dir / f"predictions_{HOLSTEIN_SUFFIX}.csv", index=False)

    output = render_error_gallery(
        runs,
        run_id,
        tmp_path / "gallery.png",
        holstein_root=holstein_root,
    )

    assert output.is_file()
    with Image.open(output) as rendered:
        assert rendered.width > 0 and rendered.height > 0
