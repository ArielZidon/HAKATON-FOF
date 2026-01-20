import firebase_admin
from firebase_admin import credentials, firestore, storage
import pandas as pd
from pathlib import Path
import os

from google.cloud import storage as gcs_storage


BASE_DIR = Path(__file__).resolve().parent

# Prefer SERVICE_KEY env var, otherwise fall back to the known key at repo root.
SERVICE_KEY = Path(
    os.environ.get(
        "SERVICE_KEY",
        str(BASE_DIR.parent.parent / "smart-fridge-c19d3-firebase-adminsdk-k1q6g-67e73c776b.json"),
    )
)
PROJECT_BUCKET = os.environ.get("PROJECT_BUCKET", "smart-fridge-c19d3.appspot.com")
BATTLEFIELD_ID = "battlefield_001"

FRIENDS_CSV_LOCAL = BASE_DIR.parent / "coordinates" / "friend.csv"
FOES_CSV_LOCAL = BASE_DIR.parent / "coordinates" / "foe.csv"


def main():
    if not SERVICE_KEY.exists():
        raise FileNotFoundError(f"Service account key not found: {SERVICE_KEY}")
    if not FRIENDS_CSV_LOCAL.exists():
        raise FileNotFoundError(f"Friends CSV not found: {FRIENDS_CSV_LOCAL}")
    if not FOES_CSV_LOCAL.exists():
        raise FileNotFoundError(f"Foes CSV not found: {FOES_CSV_LOCAL}")

    friends_dl_path = BASE_DIR / "friends_dl.csv"
    foes_dl_path = BASE_DIR / "foes_dl.csv"
    doc_ref = None

    # =========================
    # 1. אתחול Firebase
    # =========================
    cred = credentials.Certificate(str(SERVICE_KEY))
    firebase_admin.initialize_app(cred, {"storageBucket": PROJECT_BUCKET})

    # Ensure the bucket exists (some projects don't have Storage enabled yet).
    gcs_client = gcs_storage.Client(credentials=cred.get_credential(), project=cred.project_id)
    existing_bucket = gcs_client.lookup_bucket(PROJECT_BUCKET)
    if existing_bucket is None:
        raise RuntimeError(
            "Firebase Storage bucket was not found. Enable Firebase Storage in the Firebase Console "
            "or set PROJECT_BUCKET to an existing bucket name. "
            f"Bucket={PROJECT_BUCKET}."
        )

    db = firestore.client()
    bucket = storage.bucket(name=PROJECT_BUCKET)

    # =========================
    # 2. העלאת קבצים ל-Storage
    # =========================
    friends_blob = bucket.blob(f"battlefields/{BATTLEFIELD_ID}_friends.csv")
    friends_blob.upload_from_filename(str(FRIENDS_CSV_LOCAL))

    foes_blob = bucket.blob(f"battlefields/{BATTLEFIELD_ID}_foes.csv")
    foes_blob.upload_from_filename(str(FOES_CSV_LOCAL))

    print("✅ CSV files uploaded to Firebase Storage")

    # =========================
    # 3. יצירת מסמך Firestore
    # =========================
    doc_ref = db.collection("battlefields").document(BATTLEFIELD_ID)
    doc_ref.set({
        "friends_csv": f"battlefields/{BATTLEFIELD_ID}_friends.csv",
        "foes_csv": f"battlefields/{BATTLEFIELD_ID}_foes.csv",
        "status": "uploaded",
    })
    print("✅ Firestore document created")

    # Stop here: data push completed.
    print("🎉 Data push complete!")
    return


if __name__ == "__main__":
    main()