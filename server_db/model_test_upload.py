import os
from pathlib import Path

import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore


def main():
    base_dir = Path(__file__).resolve().parent

    service_key = Path(
        os.environ.get(
            "SERVICE_KEY",
            # Default: repo root service account key (override via env if needed)
            str(base_dir.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-67e73c776b.json"),
        )
    )

    csv_path = Path(
        os.environ.get(
            "CSV_PATH",
            str(base_dir.parent / "test" / "dynamic_clusters1.csv"),
        )
    )

    collection_name = os.environ.get("COLLECTION_NAME", "model_test")
    doc_id = os.environ.get("DOC_ID", "dynamic_clusters1")

    if not service_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_key}")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_key))
        firebase_admin.initialize_app(cred)

    db = firestore.client()

    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    # Store CSV content as an array of rows (records). Consider splitting into subcollection
    # if the CSV is large (Firestore doc size limit is ~1 MiB).
    payload = {
        "source_file": str(csv_path.name),
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": records,
    }

    db.collection(collection_name).document(doc_id).set(payload)

    print(
        f"✅ Uploaded {csv_path} to Firestore collection '{collection_name}' document '{doc_id}' "
        f"({len(records)} rows)"
    )


if __name__ == "__main__":
    main()
