"""
FastAPI backend for the Neuro-Fuzzy-GA Hybrid Adaptive Test Suite
Prioritization system.

Run with:
    uvicorn app:app --reload --port 8000

Endpoints:
    GET  /api/health
    GET  /api/sample?n=40          -> generates a synthetic test suite
    POST /api/upload                -> upload a CSV of tests, returns parsed rows
    POST /api/prioritize             -> body: { tests: [...], generations, population_size }
"""

import io
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sample_data import generate_sample_suite
from hybrid.pipeline import run_pipeline

app = FastAPI(title="Neuro-Fuzzy-GA CI Test Prioritizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUIRED_COLUMNS = [
    "name",
    "exec_time",
    "historical_fail_rate",
    "code_churn",
    "coverage",
    "days_since_last_run",
]

# Extra inputs for the trained TCP-CI model (all four needed to use it),
# with the value used when a cell is empty.
OPTIONAL_COLUMNS = {
    "recent_fail_rate": 0,
    "last_failure_age": -1,   # builds since the test last failed, -1 = never
    "change_coverage": 0,
    "lines_added": 0,
}


class TestCase(BaseModel):
    name: str
    exec_time: float
    historical_fail_rate: float = 0.0
    code_churn: float = 0.0
    coverage: float = 0.0
    days_since_last_run: float = 0.0
    module: Optional[str] = None
    kind: Optional[str] = None
    recent_fail_rate: Optional[float] = None
    last_failure_age: Optional[float] = None
    change_coverage: Optional[float] = None
    lines_added: Optional[float] = None


class PrioritizeRequest(BaseModel):
    tests: List[TestCase]
    generations: int = Field(default=150, ge=10, le=500)
    population_size: int = Field(default=60, ge=10, le=300)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/sample")
def sample(n: int = 40):
    if n < 1 or n > 500:
        raise HTTPException(400, "n must be between 1 and 500")
    return {"tests": generate_sample_suite(n)}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Please upload a .csv file")
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            400,
            f"CSV is missing required column(s): {', '.join(missing)}. "
            f"Required columns: {', '.join(REQUIRED_COLUMNS)}",
        )

    extra = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    for c in extra:
        df[c] = df[c].fillna(OPTIONAL_COLUMNS[c])
    df = df.fillna(0)
    tests = df[REQUIRED_COLUMNS + extra].to_dict(orient="records")
    return {"tests": tests}


@app.post("/api/prioritize")
def prioritize(req: PrioritizeRequest):
    if not req.tests:
        raise HTTPException(400, "No tests provided")
    tests = [t.model_dump() for t in req.tests]
    result = run_pipeline(tests, generations=req.generations, population_size=req.population_size)
    return result
