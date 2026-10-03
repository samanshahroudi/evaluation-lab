"""Deterministic retrieval and answer checks suitable for a local release gate."""
from __future__ import annotations

import argparse
import json
import math
import re
import time
import uuid
from collections.abc import Callable
from pathlib import Path


def score_case(case: dict, retrieve: Callable[[str, str], list[dict]]) -> dict:
    start = time.perf_counter()
    hits = retrieve(case["tenant"], case["question"])
    if not isinstance(hits, list) or any(
        not isinstance(hit, dict)
        or any(not isinstance(hit.get(field), str) or not hit[field].strip()
               for field in ("id", "tenant"))
        for hit in hits
    ):
        raise ValueError("hits must be a list of objects with nonblank string id and tenant")
    ids = [hit["id"] for hit in hits[:5] if hit["tenant"] == case["tenant"]]
    expected = set(case["relevant_ids"])
    matched = expected & set(ids[:5])
    first_rank = next((rank for rank, hit in enumerate(hits[:5], 1)
                       if hit["tenant"] == case["tenant"] and hit["id"] in expected), None)
    negative_ok = not expected and not hits
    return {"case": case["id"], "hit_at_5": bool(matched) if expected else negative_ok,
            "recall_at_5": len(matched) / len(expected) if expected else float(negative_ok),
            "reciprocal_rank": 1 / first_rank if first_rank else float(negative_ok),
            "tenant_safe": all(hit["tenant"] == case["tenant"] for hit in hits),
            "latency_ms": round((time.perf_counter() - start) * 1000, 2)}


def citation_check(answer: str, allowed_ids: set[str]) -> bool:
    """Treat every square-bracketed reference as a chunk ID and check membership."""
    cited = set(re.findall(r"\[([^\[\]]*)\]", answer))
    return bool(cited) and cited <= allowed_ids


def _validate_cases(cases: list[dict]) -> None:
    if not isinstance(cases, list) or not cases or any(
        not isinstance(case, dict) or "relevant_ids" not in case for case in cases
    ):
        raise ValueError("evaluation needs labeled cases")
    if any(not isinstance(case["relevant_ids"], list)
           or any(not isinstance(key, str) or not key.strip() for key in case["relevant_ids"])
           for case in cases):
        raise ValueError("relevant_ids must be a list of nonblank strings")
    if any(not isinstance(case.get("id"), str) or not case["id"].strip() for case in cases):
        raise ValueError("case IDs must be nonblank strings")
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("evaluation needs unique case IDs")
    for field in ("tenant", "question"):
        if any(not isinstance(case.get(field), str) or not case[field].strip() for case in cases):
            raise ValueError(f"{field} must be a nonblank string")


def evaluate(cases: list[dict], retrieve: Callable[[str, str], list[dict]]) -> dict:
    _validate_cases(cases)
    rows = [score_case(case, retrieve) for case in cases]
    return {"cases": rows, "mean_recall_at_5": sum(r["recall_at_5"] for r in rows) / len(rows),
            "mean_reciprocal_rank": sum(r["reciprocal_rank"] for r in rows) / len(rows),
            "tenant_safe": all(r["tenant_safe"] for r in rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default=str(Path(__file__).with_name("cases.json")))
    parser.add_argument("--predictions", required=True, help="JSONL rows with case_id and retrieved hits")
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--trace-file", help="Append redacted JSONL spans for local inspection")
    args = parser.parse_args()
    if not 0 <= args.min_recall <= 1:
        parser.error("--min-recall must be between 0 and 1")
    cases = json.loads(Path(args.cases).read_text())
    _validate_cases(cases)
    predictions = {}
    latencies = {}
    for line in Path(args.predictions).read_text().splitlines():
        row = json.loads(line)
        case_id = row["case_id"]
        if case_id in predictions:
            raise ValueError(f"duplicate prediction for {case_id}")
        predictions[case_id] = row["hits"]
        latency = row.get("latency_ms")
        if latency is not None and (
            isinstance(latency, bool) or not isinstance(latency, (int, float))
            or not math.isfinite(latency) or latency < 0
        ):
            raise ValueError(f"latency_ms must be a finite nonnegative number for {case_id}")
        latencies[case_id] = latency
    if set(predictions) != {case["id"] for case in cases}:
        raise ValueError("predictions must contain exactly one row per case")
    case_ids = {(case["tenant"], case["question"]): case["id"] for case in cases}
    if len(case_ids) != len(cases):
        raise ValueError("CLI cases must have unique tenant/question pairs")
    report = evaluate(cases, lambda tenant, question: predictions[case_ids[tenant, question]])
    for row in report["cases"]:
        row["latency_ms"] = latencies[row["case"]]
    if args.trace_file:
        trace_id = uuid.uuid4().hex
        with Path(args.trace_file).open("a") as handle:
            for row in report["cases"]:
                handle.write(json.dumps({"trace_id": trace_id, "event": "retrieval_evaluated",
                    "case_id": row["case"], "latency_ms": row["latency_ms"],
                    "recall_at_5": row["recall_at_5"], "tenant_safe": row["tenant_safe"]}) + "\n")
    print(json.dumps(report, indent=2))
    if report["mean_recall_at_5"] < args.min_recall or not report["tenant_safe"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
