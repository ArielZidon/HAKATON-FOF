from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import json
import time

app = FastAPI(title="Coordinate Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_FILE = Path("last_coord.json")

class Coord(BaseModel):
    x: float
    y: float

class QueryPayload(BaseModel):
    coord: Coord

@app.post("/query")
def query(payload: QueryPayload):
    x = payload.coord.x
    y = payload.coord.y

    print(f"[FROM SERVER] Received coord: x={x}, y={y}")

    data = {
        "ts": time.time(),
        "coord": {"x": x, "y": y}
    }

    # Overwrite file on every request
    with OUT_FILE.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return {
        "status": "ok",
        "message": "from server (saved to last_coord.json)",
        "saved_file": str(OUT_FILE),
        "received": {"x": x, "y": y},
    }
