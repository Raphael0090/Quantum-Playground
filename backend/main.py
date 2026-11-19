"""
main.py — FastAPI app.

This file is intentionally thin: every real branch of logic lives in
api_logic.py (which has zero FastAPI dependency and is fully unit tested).
Routes here just parse/validate the HTTP-level shape of the request,
call into api_logic, and translate ApiError -> HTTPException.

Run with:
    pip install -r requirements.txt
    python main.py
then open http://127.0.0.1:8000 in a browser.
"""
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import api_logic
from api_logic import ApiError

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Quantum Playground API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _run(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except ApiError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)


# --- Module 1: GHZ / W / Mermin ---------------------------------------------

@app.get("/api/mermin/{state_name}")
def mermin(state_name: str):
    return _run(api_logic.get_mermin, state_name)


class JointExpectationRequest(BaseModel):
    state: str
    settings: List[str] = Field(..., min_length=3, max_length=3)


@app.post("/api/joint-expectation")
def joint_expectation(body: JointExpectationRequest):
    return _run(api_logic.get_joint_expectation, body.state, body.settings)


# --- Module 2: Teleportation & superdense coding ----------------------------

@app.get("/api/teleportation")
def teleportation(theta: float = Query(...), phi: float = Query(...)):
    return _run(api_logic.get_teleportation, theta, phi)


class SuperdenseRequest(BaseModel):
    bits: str


@app.post("/api/superdense")
def superdense(body: SuperdenseRequest):
    return _run(api_logic.post_superdense, body.bits)


# --- Module 3: Density matrices ---------------------------------------------

@app.get("/api/density-matrix")
def density_matrix(kind: str = Query(...), fidelity: float = Query(...)):
    return _run(api_logic.get_density_matrix, kind, fidelity)


# --- Module 4: Tomography ----------------------------------------------------

@app.get("/api/tomography")
def tomography(
    kind: str = Query(...),
    fidelity: float = Query(...),
    shots: int = Query(...),
    seed: Optional[int] = Query(None),
):
    return _run(api_logic.get_tomography, kind, fidelity, shots, seed)


# --- Static frontend ---------------------------------------------------------

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
