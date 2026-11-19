"""
Endpoint tests using FastAPI's TestClient. These exercise the HTTP layer
(routing, request parsing, status codes) on top of the already-unit-tested
api_logic module.

Requires `fastapi` + `httpx` to be installed (see requirements.txt):
    pip install -r requirements.txt
    pytest test_api.py -v
"""
import pytest
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_mermin_ghz():
    r = client.get("/api/mermin/ghz")
    assert r.status_code == 200
    body = r.json()
    assert abs(body["M"]) == pytest.approx(4.0, abs=1e-9)


def test_mermin_w():
    r = client.get("/api/mermin/w")
    assert r.status_code == 200
    assert r.json()["M"] == pytest.approx(0.0, abs=1e-9)


def test_mermin_invalid_state():
    r = client.get("/api/mermin/bell")
    assert r.status_code == 400


def test_joint_expectation():
    r = client.post("/api/joint-expectation", json={"state": "ghz", "settings": ["X", "Y", "Y"]})
    assert r.status_code == 200
    assert abs(r.json()["expectation"]) == pytest.approx(1.0, abs=1e-9)


def test_joint_expectation_bad_settings():
    r = client.post("/api/joint-expectation", json={"state": "ghz", "settings": ["X", "Y"]})
    assert r.status_code == 422  # pydantic length validation


def test_teleportation():
    r = client.get("/api/teleportation", params={"theta": 1.1, "phi": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert len(body["outcomes"]) == 4
    for outcome in body["outcomes"]:
        assert outcome["fidelity"] == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("bits", ["00", "01", "10", "11"])
def test_superdense(bits):
    r = client.post("/api/superdense", json={"bits": bits})
    assert r.status_code == 200
    assert r.json()["success"] is True


def test_superdense_invalid_bits():
    r = client.post("/api/superdense", json={"bits": "22"})
    assert r.status_code == 400


def test_density_matrix():
    r = client.get("/api/density-matrix", params={"kind": "phi_plus", "fidelity": 1.0})
    assert r.status_code == 200
    body = r.json()
    assert body["entropy"] == pytest.approx(0.0, abs=1e-9)
    assert body["mutual_information"] == pytest.approx(2.0, abs=1e-9)


def test_density_matrix_bad_fidelity():
    r = client.get("/api/density-matrix", params={"kind": "phi_plus", "fidelity": 5.0})
    assert r.status_code == 400


@pytest.mark.parametrize("shots", [20, 200, 2000])
def test_tomography(shots):
    r = client.get("/api/tomography", params={"kind": "phi_plus", "fidelity": 0.85, "shots": shots, "seed": 1})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["hilbert_schmidt_overlap"] <= 1.0 + 1e-6


def test_tomography_bad_shots():
    r = client.get("/api/tomography", params={"kind": "phi_plus", "fidelity": 0.85, "shots": -1})
    assert r.status_code == 400


def test_index_serves_frontend():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
