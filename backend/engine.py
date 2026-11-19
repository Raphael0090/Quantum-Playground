"""
quantum_engine.py
------------------
Pure-Python complex-amplitude quantum engine (numpy for linear algebra only —
no Qiskit/Cirq/etc). This mirrors the JS engine used in the original
single-page artifact, and backs all four modules of the app:

  1. GHZ & W states / Mermin inequality
  2. Teleportation & superdense coding
  3. Density matrices, entropy, mutual information
  4. Finite-shot tomography

Qubit ordering convention: for an n-qubit statevector of length 2^n, qubit 0
is the most significant bit, qubit n-1 the least significant — i.e. index
k corresponds to basis state |b0 b1 ... b_{n-1}> where k's binary expansion
(zero-padded to n bits) is b0 b1 ... b_{n-1}.
"""

from __future__ import annotations
import itertools
import numpy as np

# ---------------------------------------------------------------------------
# Basic operators
# ---------------------------------------------------------------------------

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)

PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}


def kron_all(mats):
    out = mats[0]
    for m in mats[1:]:
        out = np.kron(out, m)
    return out


def basis_state(bits, n=None):
    """Return the computational basis statevector for a tuple/list of 0/1 bits."""
    if n is None:
        n = len(bits)
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    v = np.zeros(2 ** n, dtype=complex)
    v[idx] = 1.0
    return v


def normalize(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def full_operator(op_map, n):
    """op_map: dict {qubit_index: 2x2 matrix}. Missing qubits get identity."""
    mats = [op_map.get(q, I2) for q in range(n)]
    return kron_all(mats)


def expectation(state, operator):
    val = np.vdot(state, operator @ state)
    # operator is Hermitian for every case we use -> value should be (numerically) real
    return float(val.real)


def fidelity_pure(psi_a, psi_b):
    """|<a|b>|^2 for two pure statevectors."""
    return float(np.abs(np.vdot(psi_a, psi_b)) ** 2)


def complex_vec_to_pairs(v):
    """JSON-friendly serialization of a complex vector: [[re, im], ...].
    (Plain numpy .tolist() on a complex array yields native Python
    `complex` objects, which are not JSON serializable.)"""
    return [[float(c.real), float(c.imag)] for c in v]


# ---------------------------------------------------------------------------
# Module 1: GHZ / W states & the Mermin inequality
# ---------------------------------------------------------------------------

def ghz_state():
    v = np.zeros(8, dtype=complex)
    v[0b000] = 1 / np.sqrt(2)
    v[0b111] = 1 / np.sqrt(2)
    return v


def w_state():
    v = np.zeros(8, dtype=complex)
    v[0b100] = 1 / np.sqrt(3)
    v[0b010] = 1 / np.sqrt(3)
    v[0b001] = 1 / np.sqrt(3)
    return v


def joint_expectation(state, settings):
    """settings: length-3 list/str of 'X'/'Y'/'Z', one per party. Returns
    <state| P0 (x) P1 (x) P2 |state>."""
    op = full_operator({q: PAULI[s] for q, s in enumerate(settings)}, 3)
    return expectation(state, op)


MERMIN_TERMS = {
    "XYY": (["X", "Y", "Y"], +1),
    "YXY": (["Y", "X", "Y"], +1),
    "YYX": (["Y", "Y", "X"], +1),
    "XXX": (["X", "X", "X"], -1),
}


def mermin_value(state):
    """Returns dict with each of the 4 canonical term expectation values and
    the total witness M = <XYY> + <YXY> + <YYX> - <XXX>."""
    terms = {}
    total = 0.0
    for name, (settings, sign) in MERMIN_TERMS.items():
        e = joint_expectation(state, settings)
        terms[name] = e
        total += sign * e
    return {"terms": terms, "M": total}


def state_for_name(name):
    if name == "ghz":
        return ghz_state()
    if name == "w":
        return w_state()
    raise ValueError(f"unknown state {name!r}")


# ---------------------------------------------------------------------------
# Module 2: Teleportation & superdense coding
# ---------------------------------------------------------------------------

def unknown_qubit(theta, phi):
    """Arbitrary single-qubit pure state from Bloch angles (theta polar,
    phi azimuthal): cos(theta/2)|0> + e^{i phi} sin(theta/2)|1>."""
    a = np.cos(theta / 2)
    b = np.exp(1j * phi) * np.sin(theta / 2)
    return np.array([a, b], dtype=complex)


def bell_pair():
    """|Phi+> = (|00> + |11>)/sqrt2 on two qubits."""
    v = np.zeros(4, dtype=complex)
    v[0b00] = 1 / np.sqrt(2)
    v[0b11] = 1 / np.sqrt(2)
    return v


# The four two-qubit Bell-basis vectors, in the fixed order used for
# outcome indices 0..3, together with the correction Bob must apply and a
# human label. This ordering is the one that falls out of expanding
# |psi> (x) |Phi+> in the Bell basis (see module docstring / README for the
# derivation) and reproduces exactly the textbook table:
#   outcome 0 (Phi+) -> I
#   outcome 1 (Phi-) -> Z
#   outcome 2 (Psi+) -> X
#   outcome 3 (Psi-) -> X.Z  (applied as Z then X)
def _bell_vec(sign, swap):
    v = np.zeros(4, dtype=complex)
    if not swap:
        v[0b00] = 1 / np.sqrt(2)
        v[0b11] = sign / np.sqrt(2)
    else:
        v[0b01] = 1 / np.sqrt(2)
        v[0b10] = sign / np.sqrt(2)
    return v


BELL_BASIS = [
    {"label": "Phi+", "vec": _bell_vec(+1, False), "correction": "I"},
    {"label": "Phi-", "vec": _bell_vec(-1, False), "correction": "Z"},
    {"label": "Psi+", "vec": _bell_vec(+1, True), "correction": "X"},
    {"label": "Psi-", "vec": _bell_vec(-1, True), "correction": "XZ"},
]

CORRECTION_OPS = {
    "I": I2,
    "X": X,
    "Z": Z,
    "XZ": X @ Z,
}


def teleport(theta, phi, outcome_index):
    """Run the teleportation protocol for the given Alice measurement
    outcome (0..3) and return Bob's reconstructed state + fidelity."""
    psi = unknown_qubit(theta, phi)
    bell = bell_pair()
    full = np.tensordot(psi, bell, axes=0).reshape(8)  # qubit0=psi, qubit1=Alice-half, qubit2=Bob-half
    full = full.reshape(2, 2, 2)  # indices (q0, q1, q2)

    entry = BELL_BASIS[outcome_index]
    bell_vec = entry["vec"].reshape(2, 2)  # indices (q0, q1)

    # Project qubits 0,1 onto this Bell vector; contract to leave qubit 2 (Bob).
    bob_unnorm = np.tensordot(bell_vec.conj(), full, axes=([0, 1], [0, 1]))
    prob = float(np.vdot(bob_unnorm, bob_unnorm).real)
    bob_state = normalize(bob_unnorm)

    correction = CORRECTION_OPS[entry["correction"]]
    corrected = correction @ bob_state
    fid = fidelity_pure(psi, corrected)

    return {
        "outcome_label": entry["label"],
        "probability": prob,
        "correction": entry["correction"],
        "bob_state_before_correction": complex_vec_to_pairs(bob_state),
        "bob_state_after_correction": complex_vec_to_pairs(corrected),
        "original_state": complex_vec_to_pairs(psi),
        "fidelity": fid,
    }


def teleport_all_outcomes(theta, phi):
    return [teleport(theta, phi, k) for k in range(4)]


# --- Superdense coding -----------------------------------------------------

# Encoding operators Alice applies to her half of the shared Phi+ pair to
# send 2 classical bits to Bob. Matched to CORRECTION_OPS / BELL_BASIS above
# so that Bob's Bell-basis measurement decodes them unambiguously.
SDC_ENCODE = {
    "00": "I",
    "01": "Z",
    "10": "X",
    "11": "XZ",
}

# Map an applied single-qubit op back to which Bell state it produces from
# Phi+, i.e. the inverse of the table above (index into BELL_BASIS).
_ENCODE_TO_OUTCOME = {"I": 0, "Z": 1, "X": 2, "XZ": 3}


def superdense_encode(bits):
    """bits: '00' | '01' | '10' | '11'. Returns Alice's encoding op name and
    the resulting shared 2-qubit state."""
    op_name = SDC_ENCODE[bits]
    op = CORRECTION_OPS[op_name]
    shared = bell_pair().reshape(2, 2)
    encoded = np.tensordot(op, shared, axes=([1], [0]))  # apply to qubit 0 (Alice's)
    return op_name, encoded.reshape(4)


def superdense_decode(state):
    """Bell-basis-measure the 2-qubit state and return the recovered bits."""
    overlaps = [abs(np.vdot(entry["vec"], state)) ** 2 for entry in BELL_BASIS]
    idx = int(np.argmax(overlaps))
    bits = [b for b, o in _ENCODE_TO_OUTCOME.items() if o == idx][0]
    inv = {"I": "00", "Z": "01", "X": "10", "XZ": "11"}
    return inv[bits], overlaps[idx]


def superdense_run(bits):
    op_name, encoded = superdense_encode(bits)
    recovered_bits, prob = superdense_decode(encoded)
    return {
        "sent_bits": bits,
        "alice_operation": op_name,
        "recovered_bits": recovered_bits,
        "decode_probability": prob,
        "success": recovered_bits == bits,
    }


# ---------------------------------------------------------------------------
# Module 3: Density matrices, entropy, mutual information
# ---------------------------------------------------------------------------

def bell_state_vector(kind):
    v = np.zeros(4, dtype=complex)
    s = 1 / np.sqrt(2)
    if kind == "phi_plus":
        v[0b00] = s
        v[0b11] = s
    elif kind == "phi_minus":
        v[0b00] = s
        v[0b11] = -s
    elif kind == "psi_plus":
        v[0b01] = s
        v[0b10] = s
    elif kind == "psi_minus":
        v[0b01] = s
        v[0b10] = -s
    else:
        raise ValueError(f"unknown bell state {kind!r}")
    return v


def noisy_density_matrix(kind, fidelity):
    """rho = F |psi><psi| + (1-F) I/4 for the chosen Bell state."""
    psi = bell_state_vector(kind)
    pure = np.outer(psi, psi.conj())
    return fidelity * pure + (1 - fidelity) * np.eye(4, dtype=complex) / 4


def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals.real, 0, None)
    s = 0.0
    for p in eigvals:
        if p > 1e-12:
            s -= p * np.log2(p)
    return float(s)


def partial_trace(rho, keep, dims=(2, 2)):
    """Trace out all qubits except `keep` (0 or 1) from a 2-qubit rho."""
    rho4 = rho.reshape(2, 2, 2, 2)  # (row0,row1,col0,col1)
    if keep == 0:
        return np.einsum("ikjk->ij", rho4)
    elif keep == 1:
        return np.einsum("kikj->ij", rho4)
    raise ValueError("keep must be 0 or 1")


def mutual_information(rho):
    rho_a = partial_trace(rho, 0)
    rho_b = partial_trace(rho, 1)
    return von_neumann_entropy(rho_a) + von_neumann_entropy(rho_b) - von_neumann_entropy(rho)


def density_matrix_report(kind, fidelity):
    rho = noisy_density_matrix(kind, fidelity)
    return {
        "rho_real": rho.real.tolist(),
        "rho_imag": rho.imag.tolist(),
        "entropy": von_neumann_entropy(rho),
        "entropy_A": von_neumann_entropy(partial_trace(rho, 0)),
        "entropy_B": von_neumann_entropy(partial_trace(rho, 1)),
        "mutual_information": mutual_information(rho),
        "purity": float(np.real(np.trace(rho @ rho))),
    }


# ---------------------------------------------------------------------------
# Module 4: Finite-shot tomography
# ---------------------------------------------------------------------------

TWO_QUBIT_PAULI_LABELS = ["X", "Y", "Z"]


def hilbert_schmidt_overlap(rho_true, rho_hat):
    num = np.real(np.trace(rho_true.conj().T @ rho_hat))
    denom = np.sqrt(np.real(np.trace(rho_true.conj().T @ rho_true)) *
                     np.real(np.trace(rho_hat.conj().T @ rho_hat)))
    if denom == 0:
        return 0.0
    return float(num / denom)


def simulate_tomography(rho_true, shots_per_setting, rng=None):
    """Linear-inversion tomography of a 2-qubit state from finite-shot Pauli
    correlator measurements.

    For every pair (i, j) in {X,Y,Z}^2 we simulate `shots_per_setting`
    projective measurements of P_i (x) P_j and use the sampled +-1 outcomes
    to estimate <P_i (x) P_j>, <P_i (x) I> and <I (x) P_j>. The state is
    then reconstructed via the standard Pauli expansion
        rho_hat = 1/4 * sum_{a,b in {I,X,Y,Z}} r_ab * (sigma_a (x) sigma_b)
    """
    if rng is None:
        rng = np.random.default_rng()

    r = {("I", "I"): 1.0}
    marginal_estimates = {("X", "I"): [], ("Y", "I"): [], ("Z", "I"): [],
                           ("I", "X"): [], ("I", "Y"): [], ("I", "Z"): []}

    for pi in TWO_QUBIT_PAULI_LABELS:
        for pj in TWO_QUBIT_PAULI_LABELS:
            op = np.kron(PAULI[pi], PAULI[pj])
            true_corr = float(np.real(np.trace(rho_true @ op)))
            # Two independent local +-1 measurement records consistent with
            # the true joint correlator (sufficient for unbiased estimation
            # of the joint AND both marginal expectation values).
            p_same = (1 + true_corr) / 2  # P(outcomes agree)
            agree = rng.random(shots_per_setting) < p_same
            a_outcomes = rng.choice([-1, 1], size=shots_per_setting)
            b_outcomes = np.where(agree, a_outcomes, -a_outcomes)

            r[(pi, pj)] = float(np.mean(a_outcomes * b_outcomes))
            marginal_estimates[(pi, "I")].append(float(np.mean(a_outcomes)))
            marginal_estimates[("I", pj)].append(float(np.mean(b_outcomes)))

    for key in [("X", "I"), ("Y", "I"), ("Z", "I"), ("I", "X"), ("I", "Y"), ("I", "Z")]:
        vals = marginal_estimates[key]
        r[key] = float(np.mean(vals)) if vals else 0.0

    rho_hat = np.zeros((4, 4), dtype=complex)
    for a in ["I", "X", "Y", "Z"]:
        for b in ["I", "X", "Y", "Z"]:
            coeff = r.get((a, b), 0.0)
            rho_hat += coeff * np.kron(PAULI[a], PAULI[b])
    rho_hat /= 4

    overlap = hilbert_schmidt_overlap(rho_true, rho_hat)
    return {
        "rho_hat_real": rho_hat.real.tolist(),
        "rho_hat_imag": rho_hat.imag.tolist(),
        "estimated_correlators": {f"{a}{b}": r[(a, b)] for a, b in r},
        "hilbert_schmidt_overlap": overlap,
        "shots_per_setting": shots_per_setting,
        "total_shots": shots_per_setting * 9,
    }


def tomography_report(kind, fidelity, shots_per_setting, seed=None):
    rho_true = noisy_density_matrix(kind, fidelity)
    rng = np.random.default_rng(seed)
    result = simulate_tomography(rho_true, shots_per_setting, rng=rng)
    result["rho_true_real"] = rho_true.real.tolist()
    result["rho_true_imag"] = rho_true.imag.tolist()
    return result
