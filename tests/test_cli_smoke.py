import json

import pytest

from rossmann_forecast.cli import main


@pytest.mark.slow
def test_run_all_on_synthetic_data_writes_reports(tmp_path):
    code = main(
        [
            "run-all",
            "--synthetic",
            "--out-dir",
            str(tmp_path / "reports"),
            "--docs-dir",
            str(tmp_path / "docs"),
            "--readme",
            "none",
        ]
    )
    assert code == 0
    results = json.loads((tmp_path / "reports" / "results.json").read_text())
    assert results["all_stores"]["lightgbm"]["mean"]["wape"] > 0
    assert 0.5 < results["intervals"]["90"]["overall_coverage"] <= 1
    text = (tmp_path / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    assert "In plain English" in text and chr(0x2014) not in text


def test_check_data_missing_returns_nonzero(tmp_path, capsys):
    assert main(["check-data", "--data-dir", str(tmp_path)]) == 1
    assert "kaggle" in capsys.readouterr().out.lower()
