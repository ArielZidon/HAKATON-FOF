import json
import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, firestore


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
        out_path = out_dir / f"{doc.id}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        count += 1

    print(f"✅ Downloaded {count} documents from '{collection_name}' into: {out_dir}")


if __name__ == "__main__":
    main()
