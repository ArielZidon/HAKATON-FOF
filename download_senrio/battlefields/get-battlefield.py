import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


def _write_csv(doc_id: str, data: dict, out_dir: Path) -> Path:
    import csv

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Doc '{doc_id}' has no 'rows' array to export to CSV")

    columns = data.get("columns")
    if not isinstance(columns, list) or not columns:
        # Infer columns from union of keys (stable order)
        seen = set()
        cols = []
        for r in rows:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        cols.append(k)
        columns = cols

    out_path = out_dir / f"{doc_id}.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r if isinstance(r, dict) else {})

    return out_path


def _natural_doc_sort_key(doc_id: str):
    # Prefer numeric sort for ids like "row_12".
    # Falls back to lexicographic if no trailing number exists.
    suffix = doc_id.split("_")[-1]
    try:
        return (0, int(suffix))
    except ValueError:
        return (1, doc_id)


def main():
    base_dir = Path(__file__).resolve().parent

    service_key = Path(
        os.environ.get(
            "SERVICE_KEY",
            str(base_dir.parent.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-67e73c776b.json"),
        )
    )

    collection_name = os.environ.get("COLLECTION_NAME", "battlefields")
    out_dir = Path(os.environ.get("OUT_DIR", str(base_dir / "out")))
    sleep_seconds = float(os.environ.get("SLEEP_SECONDS", "30"))

    if not service_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_key}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_key))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    out_dir.mkdir(parents=True, exist_ok=True)

    docs = list(db.collection(collection_name).stream())
    if not docs:
        print(f"ℹ️ No documents found in collection '{collection_name}'.")
        return

    # Make output stable/predictable: row_1.csv, row_2.csv, ...
    docs.sort(key=lambda d: _natural_doc_sort_key(d.id))

    count = 0
    for doc in docs:
        data = doc.to_dict() or {}
        csv_path = _write_csv(doc.id, data, out_dir)
        count += 1
        print(f"✅ Wrote: {csv_path.name}")

        if sleep_seconds > 0 and count < len(docs):
            import time

            time.sleep(sleep_seconds)

    print(f"✅ Downloaded {count} documents from '{collection_name}' into CSV files at: {out_dir}")


if __name__ == "__main__":
    main()
