"""Keep the documented baseline and its labeled fixture IDs in sync."""
import json
import shutil
from pathlib import Path

import pytest

from evaluation_lab import seed
from evaluation_lab.evaluate import main as evaluate_main


@pytest.mark.parametrize("cases", [[], {}, [{"id": "broken"}], [{
    "id": "broken", "tenant": "demo", "question": "incident", "relevant_ids": [None],
}]])
def test_invalid_seed_case_definitions_preserve_output(tmp_path, monkeypatch, capsys, cases):
    root = tmp_path / "evaluation_lab"
    root.mkdir()
    cases_path = root / "cases.json"
    cases_path.write_text(json.dumps(cases), encoding="utf-8")
    monkeypatch.setattr(seed, "__file__", str(root / "seed.py"))
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"existing predictions\n")
    monkeypatch.setattr("sys.argv", ["seed", "--out", str(output)])
    with pytest.raises(SystemExit) as exc:
        seed.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"invalid cases file {cases_path}" in captured.err
    assert "Traceback" not in captured.err
    assert output.read_bytes() == b"existing predictions\n"


@pytest.mark.parametrize("input_name,problem", [
    ("cases.json", "missing"), ("cases.json", "directory"),
    ("cases.json", "invalid_utf8"), ("cases.json", "invalid_json"),
    ("fixtures/oncall.md", "directory"), ("fixtures/oncall.md", "invalid_utf8"),
])
def test_seed_input_errors_preserve_output(tmp_path, monkeypatch, capsys, input_name, problem):
    bundled = Path(seed.__file__).parent
    root = tmp_path / "evaluation_lab"
    shutil.copytree(bundled / "fixtures", root / "fixtures")
    shutil.copyfile(bundled / "cases.json", root / "cases.json")
    monkeypatch.setattr(seed, "__file__", str(root / "seed.py"))
    source = root / input_name
    if problem in {"missing", "directory"}:
        source.unlink()
        if problem == "directory":
            source.mkdir()
    else:
        source.write_bytes(b"\xff" if problem == "invalid_utf8" else b"{")
    output = tmp_path / "predictions.jsonl"
    output.write_bytes(b"existing predictions\n")
    monkeypatch.setattr("sys.argv", ["seed", "--out", str(output)])
    with pytest.raises(SystemExit) as exc:
        seed.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot read input file {source}" in captured.err
    assert "Traceback" not in captured.err
    assert output.read_bytes() == b"existing predictions\n"


@pytest.mark.parametrize("input_name", ["cases.json", "fixtures/oncall.md"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_seed_cannot_overwrite_its_inputs(tmp_path, monkeypatch, capsys, input_name, alias):
    bundled = Path(seed.__file__).parent
    root = tmp_path / "evaluation_lab"
    shutil.copytree(bundled / "fixtures", root / "fixtures")
    shutil.copyfile(bundled / "cases.json", root / "cases.json")
    monkeypatch.setattr(seed, "__file__", str(root / "seed.py"))
    source = root / input_name
    before = source.read_bytes()
    output = source
    if alias != "direct":
        output = tmp_path / "predictions.jsonl"
        if alias == "symlink":
            output.symlink_to(source)
        else:
            output.hardlink_to(source)
    monkeypatch.setattr("sys.argv", ["seed", "--out", str(output)])
    with pytest.raises(SystemExit) as exc:
        seed.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "predictions file must differ from cases and fixture files" in captured.err
    assert source.read_bytes() == before


@pytest.mark.parametrize("problem", ["directory", "missing_parent"])
def test_unwritable_seed_output_is_usage_error(tmp_path, monkeypatch, capsys, problem):
    output = tmp_path if problem == "directory" else tmp_path / "missing" / "predictions.jsonl"
    monkeypatch.setattr("sys.argv", ["seed", "--out", str(output)])
    with pytest.raises(SystemExit) as exc:
        seed.main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot write predictions file {output}" in captured.err
    assert "Traceback" not in captured.err
    if problem == "missing_parent":
        assert not output.exists()


def test_bundled_baseline_passes_release_gate_and_writes_redacted_traces(
    tmp_path, monkeypatch, capsys,
):
    predictions = tmp_path / "predictions.jsonl"
    traces = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["seed", "--out", str(predictions)])
    seed.main()
    capsys.readouterr()
    cases = json.loads(Path(seed.__file__).with_name("cases.json").read_text())
    rows = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["case_id"] for row in rows] == [case["id"] for case in cases]
    for case, row in zip(cases, rows):
        if not case["relevant_ids"]:
            assert row["hits"] == []
        assert all(hit["tenant"] == case["tenant"] for hit in row["hits"])

    monkeypatch.setattr("sys.argv", ["evaluate", "--predictions", str(predictions),
                                    "--min-recall", "1", "--trace-file", str(traces)])
    evaluate_main()
    report = json.loads(capsys.readouterr().out)
    assert report["mean_recall_at_5"] == 1
    assert report["mean_reciprocal_rank"] == 1
    assert report["tenant_safe"] is True
    spans = [json.loads(line) for line in traces.read_text().splitlines()]
    assert [span["case_id"] for span in spans] == [case["id"] for case in cases]
    assert len({span["trace_id"] for span in spans}) == 1
    for row, span in zip(rows, spans):
        assert span["latency_ms"] == row["latency_ms"]
        assert set(span) == {"trace_id", "event", "case_id", "latency_ms",
                             "recall_at_5", "tenant_safe"}
