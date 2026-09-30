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
