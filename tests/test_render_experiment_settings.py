from pathlib import Path
import sys


sys.path.append(str(Path(__file__).resolve().parents[1] / "tools"))
from render_experiment_settings import render_settings


def test_settings_table_is_generated_from_all_augmentations_and_training_config(tmp_path: Path):
    output = render_settings(
        "configs/cattlessfr_hardening_v2_colab_proplus.yaml",
        tmp_path / "settings.tex",
    )
    text = output.read_text(encoding="utf-8")

    assert text.count("Μετασχηματισμός &") == 20
    assert "gaussian\\_\\allowbreak{}noise" in text
    assert "cutout\\_\\allowbreak{}random" in text
    assert "learning\\_\\allowbreak{}rate\\_\\allowbreak{}finetune" in text
    assert "trainable\\_\\allowbreak{}last\\_\\allowbreak{}n" in text
    assert "early\\_\\allowbreak{}stopping\\_\\allowbreak{}patience" in text
