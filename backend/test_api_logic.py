import unittest
import api_logic
from api_logic import ApiError


class TestMerminEndpointLogic(unittest.TestCase):
    def test_ghz(self):
        r = api_logic.get_mermin("ghz")
        self.assertAlmostEqual(abs(r["M"]), 4.0, places=9)

    def test_w(self):
        r = api_logic.get_mermin("w")
        self.assertAlmostEqual(r["M"], 0.0, places=9)

    def test_invalid_state_rejected(self):
        with self.assertRaises(ApiError):
            api_logic.get_mermin("bell")

    def test_joint_expectation_validates_settings_length(self):
        with self.assertRaises(ApiError):
            api_logic.get_joint_expectation("ghz", ["X", "Y"])

    def test_joint_expectation_validates_setting_values(self):
        with self.assertRaises(ApiError):
            api_logic.get_joint_expectation("ghz", ["X", "Y", "Q"])

    def test_joint_expectation_valid(self):
        r = api_logic.get_joint_expectation("ghz", ["X", "Y", "Y"])
        self.assertAlmostEqual(abs(r["expectation"]), 1.0, places=9)


class TestTeleportationSuperdenseLogic(unittest.TestCase):
    def test_teleportation_all_outcomes_fidelity_one(self):
        r = api_logic.get_teleportation(1.1, 0.3)
        self.assertEqual(len(r["outcomes"]), 4)
        for outcome in r["outcomes"]:
            self.assertAlmostEqual(outcome["fidelity"], 1.0, places=9)

    def test_superdense_all_messages(self):
        for bits in ["00", "01", "10", "11"]:
            r = api_logic.post_superdense(bits)
            self.assertTrue(r["success"])

    def test_superdense_rejects_bad_bits(self):
        with self.assertRaises(ApiError):
            api_logic.post_superdense("22")


class TestDensityMatrixLogic(unittest.TestCase):
    def test_valid_request(self):
        r = api_logic.get_density_matrix("phi_plus", 0.9)
        self.assertIn("entropy", r)
        self.assertIn("mutual_information", r)

    def test_rejects_bad_kind(self):
        with self.assertRaises(ApiError):
            api_logic.get_density_matrix("not_a_bell_state", 0.9)

    def test_rejects_out_of_range_fidelity(self):
        with self.assertRaises(ApiError):
            api_logic.get_density_matrix("phi_plus", 1.5)


class TestTomographyLogic(unittest.TestCase):
    def test_valid_request_deterministic_with_seed(self):
        r1 = api_logic.get_tomography("phi_plus", 0.85, 200, seed=123)
        r2 = api_logic.get_tomography("phi_plus", 0.85, 200, seed=123)
        self.assertEqual(r1["hilbert_schmidt_overlap"], r2["hilbert_schmidt_overlap"])

    def test_rejects_bad_shots(self):
        with self.assertRaises(ApiError):
            api_logic.get_tomography("phi_plus", 0.85, -5)

    def test_rejects_bad_fidelity(self):
        with self.assertRaises(ApiError):
            api_logic.get_tomography("phi_plus", -0.1, 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
