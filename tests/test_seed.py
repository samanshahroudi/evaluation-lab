"""Keep the documented baseline and its labeled fixture IDs in sync."""
import json
from pathlib import Path

from evaluation_lab import seed
from evaluation_lab.evaluate import main as evaluate_main


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
