import json

import pytest

from evaluation_lab.evaluate import citation_check, evaluate


def test_gate_and_tenant_safety():
    key = "a" * 20
    cases = [{"id": "one", "tenant": "demo", "question": "incident handoff", "relevant_ids": [key]}]
    report = evaluate(cases, lambda tenant, question: [{"id": key, "tenant": tenant}])
    assert report["mean_recall_at_5"] == 1
    assert report["tenant_safe"]
    assert citation_check(f"The owner is named [{key}]", {key})
    assert not citation_check("The owner is named", {key})
    assert not citation_check(f"The owner is named [{key}] [{key}f]", {key})
    unsafe = evaluate(cases, lambda tenant, question: [{"id": key, "tenant": "other"}])
    assert not unsafe["tenant_safe"]
    assert unsafe["mean_recall_at_5"] == 0


def test_reciprocal_rank_preserves_foreign_tenant_positions():
    cases = [{"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]}]
    hits = [{"id": "a", "tenant": "other"}, {"id": "a", "tenant": "demo"}]
    report = evaluate(cases, lambda tenant, question: hits)
    assert report["mean_reciprocal_rank"] == 0.5
    assert report["mean_recall_at_5"] == 1
    assert not report["tenant_safe"]


def test_relevant_hit_beyond_top_five_is_not_scored():
    cases = [{"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]}]
    hits = [{"id": "a", "tenant": "other"}] * 5 + [{"id": "a", "tenant": "demo"}]
    report = evaluate(cases, lambda tenant, question: hits)
    assert report["mean_reciprocal_rank"] == 0
    assert report["mean_recall_at_5"] == 0


def test_citation_check_rejects_unknown_nonhex_ids():
    allowed = {"a" * 20}
    answer = f"Recovery confirmed [{'a' * 20}]"
    assert not citation_check(answer + " [unknown-source]", allowed)
    assert not citation_check(answer + " []", allowed)
    assert citation_check("Recovery confirmed [runbook-1]", {"runbook-1"})


@pytest.mark.parametrize("threshold", ["nan", "inf", "-inf", "-0.1", "1.1"])
def test_cli_rejects_invalid_recall_threshold_before_reading_files(monkeypatch, capsys, threshold):
    from evaluation_lab.evaluate import main

    monkeypatch.setattr("sys.argv", ["evaluate", "--predictions", "missing.jsonl",
                                    f"--min-recall={threshold}"])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert "--min-recall must be between 0 and 1" in capsys.readouterr().err


@pytest.mark.parametrize("threshold,exit_code", [("0", None), ("1", 1)])
def test_cli_recall_threshold_boundaries(tmp_path, monkeypatch, capsys, threshold, exit_code):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "one", "tenant": "demo", "question": "handoff",
                                 "relevant_ids": ["a"]}]))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"case_id": "one", "hits": []}) + "\n")
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--min-recall", threshold])
    if exit_code is None:
        main()
    else:
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["mean_recall_at_5"] == 0


def test_duplicate_case_ids_are_rejected_before_retrieval():
    cases = [
        {"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]},
        {"id": "one", "tenant": "demo", "question": "rollback", "relevant_ids": ["b"]},
    ]

    def unexpected_retrieval(tenant, question):
        pytest.fail("duplicate case IDs must be rejected before retrieval")

    with pytest.raises(ValueError, match="unique case IDs"):
        evaluate(cases, unexpected_retrieval)


@pytest.mark.parametrize("labels", ["", "a", None, {}, [None], [1], [""], ["   "]])
def test_invalid_relevance_labels_are_rejected_before_retrieval(labels):
    cases = [{"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": labels}]

    def unexpected_retrieval(tenant, question):
        pytest.fail("invalid labels must be rejected before retrieval")

    with pytest.raises(ValueError, match="relevant_ids must be a list of nonblank strings"):
        evaluate(cases, unexpected_retrieval)


def test_cli_rejects_ambiguous_case_lookup_without_writing_traces(tmp_path, monkeypatch, capsys):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]},
        {"id": "two", "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]},
    ]))
    predictions = tmp_path / "predictions.jsonl"
    # Reusing the second prediction would falsely give both cases perfect recall.
    predictions.write_text("\n".join(json.dumps(row) for row in [
        {"case_id": "one", "hits": []},
        {"case_id": "two", "hits": [{"id": "a", "tenant": "demo"}]},
    ]) + "\n")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(ValueError, match="unique tenant/question pairs"):
        main()
    assert capsys.readouterr().out == ""
    assert not trace.exists()


@pytest.mark.parametrize("latency", [-1, float("nan"), float("inf"), -float("inf"),
                                   "12", True])
def test_cli_rejects_invalid_latency_without_writing_traces(tmp_path, monkeypatch, capsys, latency):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "one", "tenant": "demo", "question": "handoff",
                                 "relevant_ids": ["a"]}]))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"case_id": "one", "hits": [], "latency_ms": latency}) + "\n")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(ValueError, match="latency_ms must be a finite nonnegative number"):
        main()
    assert capsys.readouterr().out == ""
    assert not trace.exists()


@pytest.mark.parametrize("latency", [None, 0, 12.5])
def test_cli_preserves_valid_optional_latency(tmp_path, monkeypatch, capsys, latency):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "one", "tenant": "demo", "question": "handoff",
                                 "relevant_ids": []}]))
    predictions = tmp_path / "predictions.jsonl"
    prediction = {"case_id": "one", "hits": []}
    if latency is not None:
        prediction["latency_ms"] = latency
    predictions.write_text(json.dumps(prediction) + "\n")
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions)])
    main()
    assert json.loads(capsys.readouterr().out)["cases"][0]["latency_ms"] == latency


@pytest.mark.parametrize("case_id", [None, "", "   ", 1, True, [], {}])
def test_invalid_case_ids_are_rejected_before_retrieval(case_id):
    cases = [{"id": case_id, "tenant": "demo", "question": "handoff", "relevant_ids": ["a"]}]

    def unexpected_retrieval(tenant, question):
        pytest.fail("invalid case IDs must be rejected before retrieval")

    with pytest.raises(ValueError, match="case IDs must be nonblank strings"):
        evaluate(cases, unexpected_retrieval)


def test_missing_case_id_is_rejected_before_retrieval():
    cases = [{"tenant": "demo", "question": "handoff", "relevant_ids": []}]
    with pytest.raises(ValueError, match="case IDs must be nonblank strings"):
        evaluate(cases, lambda tenant, question: pytest.fail("unexpected retrieval"))


@pytest.mark.parametrize("case_id", [None, "", [], {}])
def test_cli_rejects_invalid_case_ids_before_reading_predictions(tmp_path, monkeypatch, case_id):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": case_id, "tenant": "demo", "question": "handoff",
                                 "relevant_ids": []}]))
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(tmp_path / "missing.jsonl"),
                                    "--trace-file", str(trace)])
    with pytest.raises(ValueError, match="case IDs must be nonblank strings"):
        main()
    assert not trace.exists()
