"""Generate a reproducible lexical baseline for the evaluation fixture set."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path


def terms(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="predictions.jsonl")
    args = parser.parse_args()
    root = Path(__file__).parent
    cases = json.loads((root / "cases.json").read_text())
    documents = []
    for path in sorted((root / "fixtures").glob("*.md")):
        key = hashlib.sha256(f"demo:{path.name}:0".encode()).hexdigest()[:20]
        documents.append({"id": key, "tenant": "demo", "text": path.read_text()})
    documents.append({"id": "private-other-tenant", "tenant": "other", "text": "Private migration plan"})
    with Path(args.out).open("w") as output:
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
    print(f"Wrote {len(cases)} cases to {args.out}")


if __name__ == "__main__":
    main()
