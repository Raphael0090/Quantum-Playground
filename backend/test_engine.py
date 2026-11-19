"""
Test suite for the pure-Python quantum engine, mirroring the previously
verified JS behavior. Written against stdlib unittest so it runs with zero
extra dependencies (`python3 test_engine.py`); it also collects fine under
pytest if the user has it installed (`pytest test_engine.py`).
"""
import unittest
import numpy as np

import engine


class TestGhzWMermin(unittest.TestCase):
    def test_ghz_state_normalized_and_correct_amplitudes(self):
        v = engine.ghz_state()
        self.assertTrue(np.isclose(np.linalg.norm(v), 1.0))
        self.assertTrue(np.isclose(v[0b000], 1 / np.sqrt(2)))
        self.assertTrue(np.isclose(v[0b111], 1 / np.sqrt(2)))

    def test_w_state_normalized_and_correct_amplitudes(self):
        v = engine.w_state()
        self.assertTrue(np.isclose(np.linalg.norm(v), 1.0))
        for idx in (0b100, 0b010, 0b001):
            self.assertTrue(np.isclose(v[idx], 1 / np.sqrt(3)))

    def test_mermin_ghz_hits_algebraic_maximum(self):
        # GHZ saturates the Mermin bound: |M| = 4 (sign is a convention choice
        # depending on the phase/branch of GHZ used; magnitude is what's
        # physical here).
        result = engine.mermin_value(engine.ghz_state())
        self.assertAlmostEqual(abs(result["M"]), 4.0, places=9)
        for name, val in result["terms"].items():
            self.assertAlmostEqual(abs(val), 1.0, places=9, msg=name)

    def test_mermin_w_state_gives_zero(self):
        result = engine.mermin_value(engine.w_state())
        self.assertAlmostEqual(result["M"], 0.0, places=9)

    def test_joint_expectation_matches_manual_operator(self):
        state = engine.ghz_state()
        op = np.kron(np.kron(engine.X, engine.Y), engine.Y)
        expected = float(np.vdot(state, op @ state).real)
        self.assertAlmostEqual(engine.joint_expectation(state, "XYY"), expected, places=9)


class TestTeleportationSuperdense(unittest.TestCase):
    def test_teleportation_fidelity_is_always_one(self):
        cases = [(0.0, 0.0), (np.pi, 0.0), (np.pi / 2, 0.0), (np.pi / 3, 1.234), (2.1, -0.7)]
        for theta, phi in cases:
            for outcome in range(4):
                result = engine.teleport(theta, phi, outcome)
                self.assertAlmostEqual(result["fidelity"], 1.0, places=9,
                                        msg=f"theta={theta} phi={phi} outcome={outcome}")

    def test_teleportation_outcome_probabilities_are_uniform(self):
        results = engine.teleport_all_outcomes(0.9, 0.4)
        probs = [r["probability"] for r in results]
        for p in probs:
            self.assertAlmostEqual(p, 0.25, places=9)
        self.assertAlmostEqual(sum(probs), 1.0, places=9)

    def test_superdense_coding_recovers_message(self):
        for bits in ["00", "01", "10", "11"]:
            result = engine.superdense_run(bits)
            self.assertTrue(result["success"], msg=bits)
            self.assertEqual(result["recovered_bits"], bits)
            self.assertAlmostEqual(result["decode_probability"], 1.0, places=9)


class TestDensityMatrix(unittest.TestCase):
    def test_pure_bell_state_is_maximally_entangled(self):
        report = engine.density_matrix_report("phi_plus", fidelity=1.0)
        self.assertAlmostEqual(report["entropy"], 0.0, places=9)
        self.assertAlmostEqual(report["entropy_A"], 1.0, places=9)
        self.assertAlmostEqual(report["entropy_B"], 1.0, places=9)
        self.assertAlmostEqual(report["mutual_information"], 2.0, places=9)
        self.assertAlmostEqual(report["purity"], 1.0, places=9)

    def test_fully_mixed_state_has_zero_mutual_information(self):
        report = engine.density_matrix_report("phi_plus", fidelity=0.0)
        self.assertAlmostEqual(report["mutual_information"], 0.0, places=9)
        self.assertAlmostEqual(report["purity"], 0.25, places=9)

    def test_density_matrix_is_valid(self):
        rho = engine.noisy_density_matrix("psi_minus", 0.6)
        self.assertAlmostEqual(np.trace(rho).real, 1.0, places=9)
        self.assertTrue(np.allclose(rho, rho.conj().T, atol=1e-9))
        eigvals = np.linalg.eigvalsh(rho)
        self.assertTrue(np.all(eigvals > -1e-9))


class TestTomography(unittest.TestCase):
    def test_tomography_overlap_in_valid_range(self):
        rho_true = engine.noisy_density_matrix("phi_plus", 0.85)
        for shots in (20, 200, 2000):
            rng = np.random.default_rng(42)
            result = engine.simulate_tomography(rho_true, shots, rng=rng)
            self.assertGreaterEqual(result["hilbert_schmidt_overlap"], 0.0)
            self.assertLessEqual(result["hilbert_schmidt_overlap"], 1.0 + 1e-6)

    def test_tomography_overlap_high_with_many_shots(self):
        rho_true = engine.noisy_density_matrix("phi_plus", 0.85)
        rng = np.random.default_rng(7)
        result = engine.simulate_tomography(rho_true, 5000, rng=rng)
        self.assertGreater(result["hilbert_schmidt_overlap"], 0.98)

    def test_hilbert_schmidt_overlap_self_is_one(self):
        rho = engine.noisy_density_matrix("psi_plus", 0.7)
        self.assertAlmostEqual(engine.hilbert_schmidt_overlap(rho, rho), 1.0, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
