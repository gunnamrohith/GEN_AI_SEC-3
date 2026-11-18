import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

from budget import FeatureBudget, budget_checks, what_if_scenarios


ROOT = Path(__file__).resolve().parent


class TestFeatureBudget(unittest.TestCase):
    def make_budget(self, **overrides):
        budget = FeatureBudget(
            requests_per_day=1000, avg_input_tokens=1000, avg_output_tokens=500,
            peak_requests_per_minute=10, input_price_per_million=1.0,
            output_price_per_million=2.0, rpm_limit=100, tpm_limit=50000,
            monthly_budget_usd=100, daily_budget_usd=5, safety_margin_percent=20,
        )
        return replace(budget, **overrides)

    def test_cost_estimate_and_token_volumes(self):
        result = self.make_budget().calculate()
        self.assertAlmostEqual(result["cost_per_request_usd"], 0.002)
        self.assertAlmostEqual(result["daily_cost_usd"], 2.0)
        self.assertAlmostEqual(result["monthly_cost_usd"], 60.0)
        self.assertEqual(result["total_tokens_request"], 1500)
        self.assertEqual(result["input_tokens_day"], 1000000)
        self.assertEqual(result["output_tokens_day"], 500000)
        self.assertEqual(result["total_tokens_day"], 1500000)
        self.assertEqual(result["total_tokens_month"], 45000000)

    def test_capacity_passes(self):
        result = self.make_budget().calculate()
        self.assertEqual(result["peak_input_tpm"], 10000)
        self.assertEqual(result["peak_output_tpm"], 5000)
        self.assertEqual(result["peak_rpm_with_margin"], 12)
        self.assertEqual(result["peak_tpm_with_margin"], 18000)
        self.assertTrue(all(budget_checks(result).values()))

    def test_rpm_limit_fails(self):
        result = self.make_budget(rpm_limit=5).calculate()
        self.assertFalse(result["rpm_ok"])
        self.assertTrue(result["tpm_ok"])

    def test_tpm_limit_fails_independently(self):
        result = self.make_budget(tpm_limit=17000).calculate()
        self.assertFalse(result["tpm_ok"])
        self.assertTrue(result["rpm_ok"])

    def test_monthly_budget_fails(self):
        self.assertFalse(self.make_budget(monthly_budget_usd=20)
                         .calculate()["monthly_budget_ok"])

    def test_daily_budget_fails_only_after_margin(self):
        result = self.make_budget(daily_budget_usd=2.2).calculate()
        self.assertLess(result["daily_cost_usd"], 2.2)
        self.assertFalse(result["daily_budget_ok"])

    def test_document_sample_requires_monthly_review(self):
        with patch.dict(os.environ, {}, clear=True):
            result = FeatureBudget.from_env(ROOT / ".env.example").calculate()
        self.assertEqual(result["total_tokens_day"], 3000000)
        self.assertAlmostEqual(result["daily_cost_usd"], 1.44)
        self.assertAlmostEqual(result["monthly_cost_usd"], 43.20)
        self.assertAlmostEqual(result["monthly_cost_with_margin_usd"], 51.84)
        self.assertEqual(result["peak_rpm_with_margin"], 36)
        self.assertEqual(result["peak_tpm_with_margin"], 54000)
        self.assertFalse(result["monthly_budget_ok"])
        self.assertTrue(result["daily_budget_ok"])

    def test_exact_thresholds_pass_and_lower_thresholds_fail(self):
        budget = self.make_budget(safety_margin_percent=0, rpm_limit=10,
                                  tpm_limit=15000, daily_budget_usd=2,
                                  monthly_budget_usd=60, per_request_budget_usd=0.002)
        self.assertTrue(all(budget_checks(budget.calculate()).values()))
        for field, value, check in [
            ("rpm_limit", 9, "rpm_ok"), ("tpm_limit", 14999, "tpm_ok"),
            ("daily_budget_usd", 1.99, "daily_budget_ok"),
            ("monthly_budget_usd", 59.99, "monthly_budget_ok"),
            ("per_request_budget_usd", 0.00199, "per_request_budget_ok"),
        ]:
            with self.subTest(field=field):
                self.assertFalse(replace(budget, **{field: value}).calculate()[check])

    def test_optional_request_budget_uses_margin(self):
        self.assertIsNone(self.make_budget().calculate()["per_request_budget_ok"])
        result = self.make_budget(per_request_budget_usd=0.0022).calculate()
        self.assertFalse(result["per_request_budget_ok"])
        self.assertFalse(budget_checks(result)["Per-request budget"])

    def test_zero_traffic_and_zero_limits(self):
        result = self.make_budget(requests_per_day=0, peak_requests_per_minute=0,
                                  rpm_limit=0, tpm_limit=0, monthly_budget_usd=0,
                                  daily_budget_usd=0).calculate()
        self.assertEqual(result["total_tokens_day"], 0)
        self.assertEqual(result["monthly_cost_usd"], 0)
        self.assertTrue(all(budget_checks(result).values()))

    def test_negative_or_nonfinite_configuration_is_rejected(self):
        for field in fields(FeatureBudget):
            for value in (-1, float("nan"), float("inf")):
                with self.subTest(field=field.name, value=value):
                    with self.assertRaisesRegex(ValueError, field.name.upper()):
                        replace(self.make_budget(), **{field.name: value}).calculate()

    def test_fractional_counts_and_boolean_values_are_rejected(self):
        for overrides in ({"requests_per_day": 1.5}, {"daily_budget_usd": True}):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValueError):
                    self.make_budget(**overrides).calculate()

    def test_environment_overrides_and_invalid_values(self):
        for value in ("", "abc", "1.5"):
            with self.subTest(value=value):
                with patch.dict(os.environ, {"REQUESTS_PER_DAY": value}, clear=True):
                    with self.assertRaisesRegex(ValueError, "REQUESTS_PER_DAY"):
                        FeatureBudget.from_env(ROOT / ".env.example")
        with patch.dict(os.environ, {"REQUESTS_PER_DAY": "4000"}, clear=True):
            self.assertEqual(FeatureBudget.from_env(ROOT / ".env.example")
                             .requests_per_day, 4000)

    def test_missing_configuration_is_not_a_zero_cost_success(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("REQUESTS_PER_DAY=2000\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "AVG_INPUT_TOKENS is required"):
                    FeatureBudget.from_env(path)

    def test_what_if_changes_and_capacity_failure(self):
        budget = self.make_budget()
        scenarios = what_if_scenarios(budget)
        baseline = scenarios[0][1].calculate()
        doubled = scenarios[1][1].calculate()
        spike = scenarios[2][1].calculate()
        context = scenarios[3][1].calculate()
        shorter = scenarios[4][1].calculate()
        self.assertAlmostEqual(doubled["daily_cost_usd"], baseline["daily_cost_usd"] * 2)
        self.assertEqual(doubled["peak_tpm_with_margin"], baseline["peak_tpm_with_margin"])
        self.assertFalse(spike["rpm_ok"] and spike["tpm_ok"])
        self.assertGreater(context["peak_tpm_with_margin"], baseline["peak_tpm_with_margin"])
        self.assertLess(shorter["daily_cost_usd"], baseline["daily_cost_usd"])
        self.assertLess(shorter["peak_tpm_with_margin"], baseline["peak_tpm_with_margin"])
        self.assertEqual(budget.avg_output_tokens, 500)

    def test_cli_from_another_directory_and_error_path(self):
        with tempfile.TemporaryDirectory() as directory:
            environment = {
                key: value for key, value in os.environ.items()
                if key not in {field.name.upper() for field in fields(FeatureBudget)}
            }
            command = [sys.executable, str(ROOT / "budget.py")]
            result = subprocess.run(
                command + ["--env", str(ROOT / ".env.example"), "--what-if"],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=15, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Monthly budget: REVIEW", result.stdout)
            self.assertIn("Recommendation: GO", result.stdout)
            self.assertIn("Traffic spike: peak RPM -> 51", result.stdout)
            error = subprocess.run(
                command + ["--env", str(Path(directory) / "missing.env")],
                cwd=directory, env=environment, capture_output=True, text=True,
                timeout=15, check=False,
            )
            self.assertEqual(error.returncode, 2)
            self.assertIn("Configuration file not found", error.stderr)
            self.assertNotIn("Recommendation:", error.stdout)


if __name__ == "__main__":
    unittest.main()
