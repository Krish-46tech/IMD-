"""Download the public ProcessBench benchmark through the HF dataset viewer API."""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "Qwen/ProcessBench"
SPLITS = ("gsm8k", "math", "olympiadbench", "omnimath")
OUT = Path(__file__).resolve().parent / "data" / "processbench.json"


def get_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "PointOfNoReturn/1.0 research benchmark client"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main() -> None:
    rows = []
    for split in SPLITS:
        offset = 0
        while True:
            query = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split,
                                            "offset": offset, "length": 100})
            page = get_json(f"https://datasets-server.huggingface.co/rows?{query}")
            batch = [entry["row"] for entry in page["rows"]]
            rows.extend(batch)
            offset += len(batch)
            if not batch or offset >= page["num_rows_total"]:
                break
            time.sleep(0.1)
        print(f"Downloaded {split}: {offset} rows")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"Saved {len(rows)} records to {OUT}")


if __name__ == "__main__":
    main()
