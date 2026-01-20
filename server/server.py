from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Minimal Coordinate Server")

# Allow browser frontend (dev)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    return {
        "status": "ok",
        "message": "from server",
        "received": {"x": x, "y": y},
    }
