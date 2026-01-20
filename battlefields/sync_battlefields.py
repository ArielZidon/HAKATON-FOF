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
    return {
        "source_file": csv_path.name,
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


def _list_local_csvs(folder: Path, pattern: str) -> List[Path]:
    return sorted(folder.glob(pattern))


def _doc_id_for_file(csv_path: Path) -> str:
    # "battlefield_001.csv" -> "battlefield_001"
    return csv_path.stem


def sync_folder_to_collection(
    *,
    db: firestore.Client,
    folder: Path,
    pattern: str,
    collection_name: str,
    delete_missing: bool,
) -> Tuple[Set[str], Set[str], Set[str]]:
    """Sync local CSV files into a Firestore collection.

    Each CSV becomes a single Firestore document with id=<filename stem>.

    Returns: (created, updated, deleted)
    """

    local_files = _list_local_csvs(folder, pattern)
    local_ids = {_doc_id_for_file(p) for p in local_files}

    col_ref = db.collection(collection_name)

    remote_docs = list(col_ref.stream())
    remote_ids = {d.id for d in remote_docs}

    created: Set[str] = set()
    updated: Set[str] = set()
    deleted: Set[str] = set()

    for csv_path in local_files:
        doc_id = _doc_id_for_file(csv_path)
        payload = _load_csv_payload(csv_path)

        if doc_id in remote_ids:
            updated.add(doc_id)
        else:
            created.add(doc_id)

        col_ref.document(doc_id).set(payload)

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

    # Default: sync from <repo_root>/battlefields/data
    folder = Path(
        os.environ.get(
            "CSV_FOLDER",
            str(base_dir / "data"),
        )
    )

    pattern = os.environ.get("CSV_PATTERN", "*.csv")
    collection_name = os.environ.get("COLLECTION_NAME", "battlefields")

    # Mirror mode: by default, delete remote docs that don't exist locally.
    # Set DELETE_MISSING=0 to disable.
    delete_missing = os.environ.get("DELETE_MISSING", "1").strip().lower() in {"1", "true", "yes"}

    if not service_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_key}")
    if not folder.exists():
        raise FileNotFoundError(f"CSV_FOLDER not found: {folder}")

    db = _init_firebase(service_key)

    created, updated, deleted = sync_folder_to_collection(
        db=db,
        folder=folder,
        pattern=pattern,
        collection_name=collection_name,
        delete_missing=delete_missing,
    )

    print(f"✅ Synced folder={folder} pattern={pattern} collection={collection_name}")
    print(f"   created: {sorted(created)}")
    print(f"   updated: {sorted(updated)}")
    print(f"   deleted: {sorted(deleted)}" if delete_missing else "   deleted: (disabled)")


if __name__ == "__main__":
    main()
