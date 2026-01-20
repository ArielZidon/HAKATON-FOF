import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


def _write_csv(doc_id: str, data: dict, out_dir: Path) -> Path:
    """Write a Firestore document payload back to a CSV.

    Expected schema (as written by our upload/sync scripts):
      - rows: list[dict]
      - optional: columns: list[str]
      - optional: source_file: str
    """
    import csv

    rows = data.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"Doc '{doc_id}' has no 'rows' array to export to CSV")

    columns = data.get("columns")
    if not isinstance(columns, list) or not columns:
        # Infer columns from union of keys (stable order: first occurrence wins)
        seen = set()
        cols = []
        for r in rows:
            if isinstance(r, dict):
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        cols.append(k)
        columns = cols

    filename = data.get("source_file")
    if not isinstance(filename, str) or not filename.strip():
        filename = f"{doc_id}.csv"

    out_path = out_dir / filename

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r if isinstance(r, dict) else {})

    return out_path


def main():
    base_dir = Path(__file__).resolve().parent

    service_key = Path(
        os.environ.get(
            "SERVICE_KEY",
            str(base_dir.parent.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-7e57222140.json"),
        )
    )

    collection_name = os.environ.get("COLLECTION_NAME", "model_test")
    out_dir = Path(os.environ.get("OUT_DIR", str(base_dir / "out")))
    write_json = os.environ.get("WRITE_JSON", "0").strip() in {"1", "true", "True"}

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

    count = 0
    for doc in docs:
        data = doc.to_dict() or {}

        csv_path = _write_csv(doc.id, data, out_dir)

        if write_json:
            json_path = out_dir / f"{doc.id}.json"
            json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        count += 1

    print(f"✅ Downloaded {count} documents from '{collection_name}' into CSV files at: {out_dir}")


if __name__ == "__main__":
    main()
