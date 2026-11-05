"""Upload the policy documents in data/corpus through the API.

    python scripts/seed.py                      # API at http://localhost:8000
    python scripts/seed.py --api http://host:8000

Re-running is safe, because the API ignores duplicates.
"""

import argparse
from pathlib import Path

import httpx

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()
    files = sorted(CORPUS.glob("*.pdf"))
    if not files:
        raise SystemExit(f"No PDFs found in {CORPUS}")
    with httpx.Client(timeout=120) as client:
        for path in files:
            with path.open("rb") as f:
                r = client.post(f"{args.api}/api/documents", params={"title": path.stem}, files={"file": (path.name, f)})
            r.raise_for_status()
            print(path.name, r.json())


if __name__ == "__main__":
    main()
