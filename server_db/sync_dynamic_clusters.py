import os
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore


def _init_firebase(service_key: Path) -> firestore.Client:
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_key))
        firebase_admin.initialize_app(cred)
    return firestore.client()


def _load_csv_payload(csv_path: Path) -> Dict:
    df = pd.read_csv(csv_path)
    records = df.to_dict(orient="records")

    return {
        "source_file": csv_path.name,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": records,
    }


def _list_local_csvs(folder: Path, pattern: str) -> List[Path]:
    return sorted(folder.glob(pattern))


def _doc_id_for_file(csv_path: Path) -> str:
    # "dynamic_clusters1.csv" -> "dynamic_clusters1"
    return csv_path.stem


def sync_dynamic_clusters(
    *,
    db: firestore.Client,
    folder: Path,
    pattern: str,
    collection_name: str,
    delete_missing: bool,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Synchronize local dynamic_clusters*.csv files into a Firestore collection.

    Returns: (created, updated, deleted) document id sets.
    """

    local_files = _list_local_csvs(folder, pattern)
    local_ids = {_doc_id_for_file(p) for p in local_files}

    col_ref = db.collection(collection_name)

    # Fetch existing doc ids in the collection
    remote_docs = list(col_ref.stream())
    remote_ids = {d.id for d in remote_docs}

    created: Set[str] = set()
    updated: Set[str] = set()
    deleted: Set[str] = set()

    # Upsert local files
    for csv_path in local_files:
        doc_id = _doc_id_for_file(csv_path)
        payload = _load_csv_payload(csv_path)

        # Heuristic: treat as created if doc doesn't exist.
        if doc_id in remote_ids:
            updated.add(doc_id)
        else:
            created.add(doc_id)

        col_ref.document(doc_id).set(payload)

    # Optionally delete docs that no longer exist locally
    if delete_missing:
        for doc_id in sorted(remote_ids - local_ids):
            col_ref.document(doc_id).delete()
            deleted.add(doc_id)

    return created, updated, deleted


def main():
    base_dir = Path(__file__).resolve().parent

    service_key = Path(
        os.environ.get(
            "SERVICE_KEY",
            str(base_dir.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-67e73c776b.json"),
        )
    )

    folder = Path(
        os.environ.get(
            "CSV_FOLDER",
            str(base_dir.parent / "test"),
        )
    )

    pattern = os.environ.get("CSV_PATTERN", "dynamic_clusters*.csv")
    collection_name = os.environ.get("COLLECTION_NAME", "model_test")

    # Safety: default is NOT deleting anything remotely unless explicitly enabled.
    delete_missing = os.environ.get("DELETE_MISSING", "0").strip().lower() in {"1", "true", "yes"}

    if not service_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_key}")
    if not folder.exists():
        raise FileNotFoundError(f"CSV_FOLDER not found: {folder}")

    db = _init_firebase(service_key)

    created, updated, deleted = sync_dynamic_clusters(
        db=db,
        folder=folder,
        pattern=pattern,
        collection_name=collection_name,
        delete_missing=delete_missing,
    )

    print(f"✅ Synced folder={folder} pattern={pattern} collection={collection_name}")
    print(f"   created: {sorted(created)}")
    print(f"   updated: {sorted(updated)}")
    if delete_missing:
        print(f"   deleted: {sorted(deleted)}")
    else:
        print("   deleted: (disabled) set DELETE_MISSING=1 to enable")


if __name__ == "__main__":
    main()
