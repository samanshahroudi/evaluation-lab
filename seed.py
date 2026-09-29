"""Build the deterministic fixture database used by the evaluation examples."""
import argparse
from importlib import import_module
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="eval.db")
    args = parser.parse_args()
    Store = import_module("02_knowledge_retrieval.retrieval").Store
    store = Store(args.db)
    fixtures = Path(__file__).parents[1] / "02_knowledge_retrieval" / "fixtures"
    for path in sorted(fixtures.glob("*.md")):
        store.ingest("demo", path.name, path.read_text())
    store.ingest("other", "private.md", "Private customer migration plan and access keys")
    print("Seeded demo and other tenant corpora")


if __name__ == "__main__":
    main()
