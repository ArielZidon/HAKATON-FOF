from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import subprocess
import sys

app = FastAPI(title="Inference Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

class Coord(BaseModel):
    x: float  # longitude (lng)
    y: float  # latitude  (lat)

class QueryPayload(BaseModel):
    coord: Coord

@app.post("/query")
def query(payload: QueryPayload):
    longitude = payload.coord.x
    latitude = payload.coord.y

    cmd = [
        sys.executable, str(BASE_DIR / "inference.py"),
        "--model-path", str(BASE_DIR / "models" / "trained_model.pkl"),
        "--state-files",
        str(BASE_DIR / "coordinates" / "field1.csv"),
        str(BASE_DIR / "coordinates" / "field2.csv"),
        str(BASE_DIR / "coordinates" / "field3.csv"),
        "--longitude", str(longitude),
        "--latitude", str(latitude),
    ]

    p = subprocess.run(
        cmd,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True
    )
    
    # Print on server terminal
    print("[RUN]", " ".join(cmd))
    print("[STDOUT]\n", p.stdout)
    if p.stderr:
        print("[STDERR]\n", p.stderr)

    return {
        "returncode": p.returncode,
        "stdout": p.stdout,
        "stderr": p.stderr,
    }
