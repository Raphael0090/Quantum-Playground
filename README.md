# Quantum Playground — Python edition

A four-module quantum mechanics playground with a numpy-backed FastAPI
backend and a vanilla-JS frontend. Same physics engine that powered the
original single-page JS artifact, ported to Python and re-verified.

## Modules

1. **GHZ & W states / Mermin inequality** — toggle each of three parties
   between X/Y measurement settings and watch
   `M = ⟨XYY⟩ + ⟨YXY⟩ + ⟨YYX⟩ − ⟨XXX⟩`. GHZ saturates the algebraic bound
   `|M| = 4`; W gives exactly 0 on this witness (it's entangled — this
   particular test just isn't built to see it).
2. **Teleportation & superdense coding** — set an arbitrary unknown qubit
   via Bloch angles θ/φ, step through all four of Alice's possible
   measurement outcomes, and watch Bob's correction (I / Z / X / XZ)
   reconstruct it with fidelity exactly 1.0000 every time. Superdense
   coding runs the shared pair in reverse and always recovers Alice's
   2-bit message.
3. **Density matrix & entropy** — the full 4×4
   `ρ = F|ψ⟩⟨ψ| + (1−F)·I/4` for any Bell state and noise level, its von
   Neumann entropy, marginal entropies, and mutual information.
4. **Finite-shot tomography** — simulate 20 / 200 / 2000 shots per Pauli
   setting and reconstruct ρ̂ via linear inversion, compared to the truth
   through a Hilbert–Schmidt overlap.

## Project layout

```
quantum_app/
├── backend/
│   ├── engine.py          — pure-Python/numpy quantum engine (no QC frameworks)
│   ├── api_logic.py       — framework-free request/response logic
│   ├── main.py            — FastAPI app (thin wrapper over api_logic)
│   ├── test_engine.py     — 14 unit tests for the engine
│   ├── test_api_logic.py  — 15 unit tests for the API logic layer
│   ├── test_api.py        — FastAPI TestClient endpoint tests
│   └── requirements.txt
└── frontend/
    └── index.html         — single-page app, four tabs, talks to /api/*
```

## Running it

```bash
cd backend
pip install -r requirements.txt
python main.py
```

Then open **http://127.0.0.1:8000** in a browser. The FastAPI app serves
the frontend directly, so that's the only URL you need.

## Running the tests

```bash
cd backend
python3 -m unittest test_engine test_api_logic -v   # zero extra deps
pytest test_api.py -v                                # needs fastapi + httpx
```

## A note on how this was verified

Development happened in a sandboxed environment with no network access, so
`fastapi` itself couldn't be installed or executed there — `engine.py` and
`api_logic.py` (which hold 100% of the actual logic) were fully unit tested
directly, and `main.py` is a deliberately thin, standard FastAPI wrapper
around them. To verify the full stack end-to-end anyway, a temporary Flask
mirror (not shipped) called the *exact same* `api_logic.py` functions
`main.py` calls, and the real, unmodified `frontend/index.html` was driven
against it with a headless-Chromium (Playwright) test that actually clicked
buttons and dragged sliders — confirming GHZ hits `|M|=4`, W gives `M=0`,
teleportation fidelity is `1.0000` across all four outcomes even after
changing θ, superdense coding recovers all four bit patterns, and
tomography overlap climbs toward 1 with more shots. That run caught and
fixed a real bug (`numpy` complex arrays don't JSON-serialize via
`.tolist()` the way you'd expect — see `engine.complex_vec_to_pairs`).

If you hit anything unexpected once `fastapi` is actually installed and
running, it's most likely in the thin `main.py` routing layer rather than
the engine — that's the one piece that couldn't be executed pre-delivery.
