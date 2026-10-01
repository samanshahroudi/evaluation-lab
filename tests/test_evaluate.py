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
