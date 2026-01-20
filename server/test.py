import os
from pathlib import Path

import firebase_admin
from firebase_admin import credentials, storage


def main():
    base_dir = Path(__file__).resolve().parent

    service_key = Path(
        os.environ.get(
            "SERVICE_KEY",
            str(base_dir.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-67e73c776b.json"),
        )
    )
    project_bucket = os.environ.get("PROJECT_BUCKET", "smart-fridge-c19d3.appspot.com")
    battlefield_id = os.environ.get("BATTLEFIELD_ID", "battlefield_001")

    if not service_key.exists():
        raise FileNotFoundError(f"Service account key not found: {service_key}")

    # Initialize Firebase Admin once.
    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_key))
        firebase_admin.initialize_app(cred, {"storageBucket": project_bucket})

    bucket = storage.bucket(name=project_bucket)

    friends_blob_path = f"battlefields/{battlefield_id}_friends.csv"
    foes_blob_path = f"battlefields/{battlefield_id}_foes.csv"

    friends_local = base_dir / "friends_downloaded.csv"
    foes_local = base_dir / "foes_downloaded.csv"

    bucket.blob(friends_blob_path).download_to_filename(str(friends_local))
    bucket.blob(foes_blob_path).download_to_filename(str(foes_local))

    print(f"✅ Downloaded: gs://{project_bucket}/{friends_blob_path} -> {friends_local}")
    print(f"✅ Downloaded: gs://{project_bucket}/{foes_blob_path} -> {foes_local}")

    print("\n===== FRIENDS CSV =====")
    print(friends_local.read_text(encoding="utf-8", errors="replace"))

    print("\n===== FOES CSV =====")
    print(foes_local.read_text(encoding="utf-8", errors="replace"))


if __name__ == "__main__":
    main()
