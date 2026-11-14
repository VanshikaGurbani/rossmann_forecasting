"""Execute every code cell of the notebooks so they cannot silently break."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
HAS_DATA = (ROOT / "data" / "train.csv").exists() and (ROOT / "data" / "store.csv").exists()


def test_the_notebook_set_is_complete():
    names = [p.name for p in NOTEBOOKS]
    assert len(names) == 8, names
    assert names[0].startswith("00_") and names[-1].startswith("07_")


def test_saved_outputs_contain_no_errors():
    for path in NOTEBOOKS:
        nb = json.loads(path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            for out in cell.get("outputs", []):
                assert out["output_type"] != "error", f"{path.name} has a saved error"


@pytest.mark.slow
@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs_end_to_end(path, monkeypatch):
    if not path.name.startswith("00_") and not HAS_DATA:
        pytest.skip("needs data/train.csv and data/store.csv (see README)")
    import matplotlib

    matplotlib.use("Agg")
    nb = json.loads(path.read_text(encoding="utf-8"))
    cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    source = "\n".join("".join(c["source"]) for c in cells)
    monkeypatch.chdir(path.parent)
    exec(compile(source, path.name, "exec"), {"__name__": "__main__"})
