# Add this route to your existing api.py / provider router.
# This is what receives feature+label pairs from Android edge devices —
# it never receives raw SMS text.

from fastapi import APIRouter
from pydantic import BaseModel
import json
import sqlite3

router = APIRouter(prefix="/edge", tags=["edge-sync"])


class FeedbackPayload(BaseModel):
    featuresJson: str   # e.g. "[2.0, 87.0, 1.0, 1.0]"
    label: int          # 1 = confirmed smishing, 0 = false positive
    riskScore: float
    deviceId: str


@router.post("/feedback")
def receive_feedback(payload: FeedbackPayload):
    """
    Stores labeled feature vectors from edge devices into the Provider's
    local training pool. Provider periodically batches these into
    cohort_update() (provider/provider_training.py) rather than training
    on each submission individually.
    """
    features = json.loads(payload.featuresJson)

    conn = sqlite3.connect("provider_training_pool.db")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS training_pool (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            features_json TEXT NOT NULL,
            label INTEGER NOT NULL,
            risk_score REAL,
            device_id TEXT,
            consumed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute(
        "INSERT INTO training_pool (features_json, label, risk_score, device_id) VALUES (?, ?, ?, ?)",
        (payload.featuresJson, payload.label, payload.riskScore, payload.deviceId)
    )
    conn.commit()
    conn.close()

    return {"status": "received"}
