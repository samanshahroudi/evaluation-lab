"""Generate a reproducible lexical baseline for the evaluation fixture set."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

from .evaluate import _validate_cases


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="predictions.jsonl")
    args = parser.parse_args()
    root = Path(__file__).parent
    cases_path = root / "cases.json"
    fixture_paths = sorted((root / "fixtures").glob("*.md"))
    try:
        cases = json.loads(cases_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read input file {cases_path}: {exc}")
    try:
        _validate_cases(cases)
    except ValueError as exc:
        parser.error(f"invalid cases file {cases_path}: {exc}")
    documents = []
    for path in fixture_paths:
        key = hashlib.sha256(f"demo:{path.name}:0".encode()).hexdigest()[:20]
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            parser.error(f"cannot read input file {path}: {exc}")
        documents.append({"id": key, "tenant": "demo", "text": text})
    documents.append({"id": "private-other-tenant", "tenant": "other", "text": "Private migration plan"})
    try:
        output_path = Path(args.out)
        if output_path.exists() and any(
            output_path.samefile(path) for path in (cases_path, *fixture_paths)
        ):
            parser.error("predictions file must differ from cases and fixture files")
        with output_path.open("w", encoding="utf-8") as output:
            for case in cases:
                start = time.perf_counter()
                query_terms = terms(case["question"])
                ranked = sorted(((len(query_terms & terms(doc["text"])), doc)
                                 for doc in documents if doc["tenant"] == case["tenant"]),
                                key=lambda pair: pair[0], reverse=True)
                hits = [{"id": doc["id"], "tenant": doc["tenant"]}
                        for score, doc in ranked if score > 0][:5]
                output.write(json.dumps({"case_id": case["id"], "hits": hits,
                                         "latency_ms": round((time.perf_counter() - start) * 1000, 2)}) + "\n")
    except OSError as exc:
        parser.error(f"cannot write predictions file {args.out}: {exc}")
    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
