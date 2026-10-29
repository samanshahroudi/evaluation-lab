import json

import pytest

from evaluation_lab.evaluate import citation_check, evaluate


def test_cli_malformed_hits_identify_prediction_line_and_case(tmp_path, monkeypatch, capsys):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "one", "tenant": "demo", "question": "first", "relevant_ids": []},
        {"id": "two", "tenant": "demo", "question": "second", "relevant_ids": []},
    ]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"case_id": "one", "hits": []}\n'
        '{"case_id": "two", "hits": [{"id": "chunk", "tenant": " "}]}\n',
        encoding="utf-8",
    )
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(predictions) in captured.err
    assert "line 2" in captured.err
    assert "case two" in captured.err
    assert "hits must be a list of objects" in captured.err
    assert not trace.exists()


def test_successful_cli_trace_appends_correlated_redacted_spans(tmp_path, monkeypatch, capsys):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([
        {"id": "answerable", "tenant": "demo", "question": "sensitive question",
         "relevant_ids": ["private-chunk"]},
        {"id": "unanswerable", "tenant": "demo", "question": "another sensitive question",
         "relevant_ids": []},
    ]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("\n".join(json.dumps(row) for row in [
        {"case_id": "answerable", "hits": [{"id": "private-chunk", "tenant": "demo",
                                           "text": "private document text"}], "latency_ms": 12.5},
        {"case_id": "unanswerable", "hits": []},
    ]) + "\n", encoding="utf-8")
    trace = tmp_path / "traces.jsonl"
    existing = '{"event": "previous-run"}\n'
    trace.write_text(existing, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    main()
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["mean_recall_at_5"] == 1
    contents = trace.read_text(encoding="utf-8")
    assert contents.startswith(existing)
    spans = [json.loads(line) for line in contents.splitlines()[1:]]
    assert len(spans) == 2
    trace_id = spans[0]["trace_id"]
    assert len(trace_id) == 32
    for span, case_id, latency in zip(spans, ["answerable", "unanswerable"], [12.5, None]):
        assert span == {
            "trace_id": trace_id, "event": "retrieval_evaluated", "case_id": case_id,
            "latency_ms": latency, "recall_at_5": 1, "tenant_safe": True,
        }
    for sensitive in ("sensitive question", "private-chunk", "private document text"):
        assert sensitive not in contents


@pytest.mark.parametrize("input_name", ["cases", "predictions"])
@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_trace_destination_cannot_modify_inputs(
    tmp_path, monkeypatch, capsys, input_name, alias,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": [],
    }]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"case_id": "one", "hits": []}\n', encoding="utf-8")
    before = {path: path.read_bytes() for path in (cases, predictions)}
    source = cases if input_name == "cases" else predictions
    trace = source if alias == "direct" else tmp_path / "traces.jsonl"
    if alias == "symlink":
        trace.symlink_to(source)
    elif alias == "hardlink":
        trace.hardlink_to(source)
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "trace file must differ from cases and predictions files" in captured.err
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("foreign_rank", [1, 6])
def test_cli_tenant_leak_fails_gate_even_with_perfect_recall(
    tmp_path, monkeypatch, capsys, foreign_rank,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    predictions = tmp_path / "predictions.jsonl"
    trace = tmp_path / "traces.jsonl"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "outage", "relevant_ids": ["relevant"],
    }]), encoding="utf-8")
    hits = [{"id": "relevant", "tenant": "demo"},
            *({"id": f"other-{index}", "tenant": "demo"} for index in range(4))]
    hits.insert(foreign_rank - 1, {"id": "private", "tenant": "other"})
    predictions.write_text(json.dumps({"case_id": "one", "hits": hits}) + "\n",
                           encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--min-recall", "1",
                                    "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    report = json.loads(captured.out)
    assert report["mean_recall_at_5"] == 1
    assert report["tenant_safe"] is False
    span = json.loads(trace.read_text(encoding="utf-8"))
    assert span["tenant_safe"] is False
    assert span["recall_at_5"] == 1
    assert "private" not in trace.read_text(encoding="utf-8")


@pytest.mark.parametrize("input_file", ["cases", "predictions"])
@pytest.mark.parametrize("problem", ["missing", "directory", "invalid_utf8", "invalid_json"])
def test_cli_input_file_errors_are_usage_errors_without_output(
    tmp_path, monkeypatch, capsys, input_file, problem,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    predictions = tmp_path / "predictions.jsonl"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": [],
    }]), encoding="utf-8")
    predictions.write_text('{"case_id": "one", "hits": []}\n', encoding="utf-8")
    broken = cases if input_file == "cases" else predictions
    if problem in ("missing", "directory"):
        broken.unlink()
        if problem == "directory":
            broken.mkdir()
    elif problem == "invalid_utf8":
        broken.write_bytes(b"\xff")
    else:
        broken.write_text('{"case_id": "one", "hits": []}\n{broken' if input_file == "predictions"
                          else '{broken', encoding="utf-8")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(broken) in captured.err
    assert "error:" in captured.err
    assert "Traceback" not in captured.err
    if input_file == "predictions" and problem == "invalid_json":
        assert "line 2" in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("answer", ["[[runbook-1]]", "[outer [runbook-1]]",
                                    "[runbook-1] [unfinished", "[runbook-1] stray]"])
def test_citation_check_rejects_malformed_brackets(answer):
    assert not citation_check(answer, {"runbook-1"})


def test_cli_duplicate_predictions_are_usage_errors_without_report_or_traces(
    tmp_path, monkeypatch, capsys,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": [],
    }]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"case_id": "one", "hits": []}\n' * 2, encoding="utf-8")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 2: duplicate prediction for one" in captured.err
    assert "Traceback" not in captured.err
    assert not trace.exists()


def test_citation_check_accepts_repeated_and_multiple_valid_references():
    assert citation_check("Recovery [runbook-1] and handoff [handoff-2] [runbook-1].",
                          {"runbook-1", "handoff-2"})


@pytest.mark.parametrize("field", ["tenant", "question"])
@pytest.mark.parametrize("value", [None, "", "   ", 1, True, [], {}])
def test_invalid_retrieval_inputs_are_rejected_before_retrieval(field, value):
    cases = [{"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": []},
             {"id": "two", "tenant": "demo", "question": "rollback", "relevant_ids": []}]
    cases[1][field] = value
    with pytest.raises(ValueError, match=f"{field} must be a nonblank string"):
        evaluate(cases, lambda tenant, question: pytest.fail("unexpected retrieval"))


@pytest.mark.parametrize("field", ["tenant", "question"])
def test_missing_retrieval_inputs_are_rejected_before_retrieval(field):
    case = {"id": "one", "tenant": "demo", "question": "handoff", "relevant_ids": []}
    del case[field]
    with pytest.raises(ValueError, match=f"{field} must be a nonblank string"):
        evaluate([case], lambda tenant, question: pytest.fail("unexpected retrieval"))


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
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "unique tenant/question pairs" in captured.err
    assert "Traceback" not in captured.err
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
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "line 1" in captured.err
    assert "one" in captured.err
    assert "latency_ms must be a finite nonnegative number" in captured.err
    assert "Traceback" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("latency", [None, 0, 12.5,
                                   pytest.param(10**400, id="large-integer")])
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
def test_cli_rejects_invalid_case_ids_before_reading_predictions(tmp_path, monkeypatch, capsys, case_id):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": case_id, "tenant": "demo", "question": "handoff",
                                 "relevant_ids": []}]))
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(tmp_path / "missing.jsonl"),
                                    "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "case IDs must be nonblank strings" in captured.err
    assert "cannot read predictions" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("hits", ["", None, {}, [None], [{}],
                                 [{"id": "a", "tenant": " "}],
                                 [{"id": 1, "tenant": "demo"}]])
def test_malformed_hits_cannot_pass_unanswerable_case(hits):
    cases = [{"id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": []}]
    with pytest.raises(ValueError, match="hits must be a list of objects"):
        evaluate(cases, lambda tenant, question: hits)


def test_cli_rejects_string_hits_without_report_or_traces(tmp_path, monkeypatch, capsys):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "one", "tenant": "demo", "question": "unknown",
                                 "relevant_ids": []}]))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps({"case_id": "one", "hits": ""}) + "\n")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hits must be a list of objects" in captured.err
    assert "Traceback" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("cases", [None, 1, True, "relevant_ids", {},
                                 {"one": {"relevant_ids": []}}, [],
                                 [None], [1], [True], ["relevant_ids"], [[]]])
def test_malformed_case_containers_are_rejected_before_retrieval(cases):
    with pytest.raises(ValueError, match="evaluation needs labeled cases"):
        evaluate(cases, lambda tenant, question: pytest.fail("unexpected retrieval"))


def test_cli_rejects_nonobject_cases_before_reading_predictions(tmp_path, monkeypatch, capsys):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text("[null]")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(tmp_path / "missing.jsonl"),
                                    "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "evaluation needs labeled cases" in captured.err
    assert "cannot read predictions" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("prediction", [None, [], "row", 1, {},
                                       {"case_id": "one"}, {"case_id": [], "hits": []},
                                       {"case_id": " ", "hits": []}])
def test_cli_rejects_malformed_prediction_rows(tmp_path, monkeypatch, capsys, prediction):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{"id": "one", "tenant": "demo", "question": "unknown",
                                 "relevant_ids": []}]))
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(json.dumps(prediction) + "\n")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid prediction at line 1" in captured.err
    assert "prediction rows need a nonblank string case_id and hits" in captured.err
    assert "Traceback" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("prediction_ids", [[], ["other"], ["one", "other"]])
def test_cli_prediction_case_mismatch_is_usage_error_without_output(
    tmp_path, monkeypatch, capsys, prediction_ids,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": [],
    }]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text("".join(
        json.dumps({"case_id": key, "hits": []}) + "\n" for key in prediction_ids
    ), encoding="utf-8")
    trace = tmp_path / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "predictions must contain exactly one row per case" in captured.err
    assert "Traceback" not in captured.err
    assert not trace.exists()


@pytest.mark.parametrize("problem", ["directory", "missing_parent"])
def test_unwritable_trace_file_is_usage_error_without_report(
    tmp_path, monkeypatch, capsys, problem,
):
    from evaluation_lab.evaluate import main

    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([{
        "id": "one", "tenant": "demo", "question": "unknown", "relevant_ids": [],
    }]), encoding="utf-8")
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text('{"case_id": "one", "hits": []}\n', encoding="utf-8")
    trace = tmp_path if problem == "directory" else tmp_path / "missing" / "traces.jsonl"
    monkeypatch.setattr("sys.argv", ["evaluate", "--cases", str(cases),
                                    "--predictions", str(predictions), "--trace-file", str(trace)])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"cannot write trace file {trace}" in captured.err
    assert "Traceback" not in captured.err
    assert predictions.read_text(encoding="utf-8") == '{"case_id": "one", "hits": []}\n'
