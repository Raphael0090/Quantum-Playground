"""
api_logic.py
------------
Pure-Python request/response handlers: plain functions taking primitive
arguments and returning JSON-serializable dicts. Deliberately has ZERO
dependency on FastAPI/Flask/etc, so it can be fully unit tested on its own.
main.py (the actual web app) is a thin routing layer on top of this module —
every branch of real logic lives here, not there.
"""

from __future__ import annotations
import engine


class ApiError(Exception):
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


VALID_BELL_KINDS = {"phi_plus", "phi_minus", "psi_plus", "psi_minus"}
VALID_SETTINGS = {"X", "Y", "Z"}


def _require(cond, message):
    if not cond:
        raise ApiError(message)


# --- Module 1: GHZ / W / Mermin --------------------------------------------

def get_mermin(state_name):
    _require(state_name in ("ghz", "w"), f"state must be 'ghz' or 'w', got {state_name!r}")
    state = engine.state_for_name(state_name)
    result = engine.mermin_value(state)
    return {
        "state": state_name,
        "terms": result["terms"],
        "M": result["M"],
        "algebraic_bound": 4.0,
    }


def get_joint_expectation(state_name, settings):
    _require(state_name in ("ghz", "w"), f"state must be 'ghz' or 'w', got {state_name!r}")
    _require(isinstance(settings, (list, tuple)) and len(settings) == 3,
              "settings must be a list of 3 measurement bases")
    for s in settings:
        _require(s in VALID_SETTINGS, f"invalid measurement setting {s!r}")
    state = engine.state_for_name(state_name)
    value = engine.joint_expectation(state, settings)
    return {"state": state_name, "settings": list(settings), "expectation": value}


# --- Module 2: Teleportation & superdense coding ---------------------------

def get_teleportation(theta, phi):
    _require(isinstance(theta, (int, float)), "theta must be a number")
    _require(isinstance(phi, (int, float)), "phi must be a number")
    results = engine.teleport_all_outcomes(theta, phi)
    return {
        "theta": theta,
        "phi": phi,
        "original_state": engine.complex_vec_to_pairs(engine.unknown_qubit(theta, phi)),
        "outcomes": results,
    }


def post_superdense(bits):
    _require(bits in ("00", "01", "10", "11"), f"bits must be one of 00/01/10/11, got {bits!r}")
    return engine.superdense_run(bits)


# --- Module 3: Density matrices ---------------------------------------------

def get_density_matrix(kind, fidelity):
    _require(kind in VALID_BELL_KINDS, f"kind must be one of {sorted(VALID_BELL_KINDS)}, got {kind!r}")
    _require(isinstance(fidelity, (int, float)) and 0.0 <= fidelity <= 1.0,
              "fidelity must be a number in [0, 1]")
    report = engine.density_matrix_report(kind, fidelity)
    report["kind"] = kind
    report["fidelity"] = fidelity
    return report


# --- Module 4: Tomography ---------------------------------------------------

VALID_SHOT_COUNTS = (20, 200, 2000)


def get_tomography(kind, fidelity, shots_per_setting, seed=None):
    _require(kind in VALID_BELL_KINDS, f"kind must be one of {sorted(VALID_BELL_KINDS)}, got {kind!r}")
    _require(isinstance(fidelity, (int, float)) and 0.0 <= fidelity <= 1.0,
              "fidelity must be a number in [0, 1]")
    _require(isinstance(shots_per_setting, int) and shots_per_setting > 0,
              "shots_per_setting must be a positive integer")
    report = engine.tomography_report(kind, fidelity, shots_per_setting, seed=seed)
    report["kind"] = kind
    report["fidelity"] = fidelity
    return report
