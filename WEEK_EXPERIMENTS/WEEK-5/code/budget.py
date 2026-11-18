"""Offline token cost and rate-limit planner for the Week 5 lab."""

import argparse
import math
import os
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import TypedDict

from dotenv import dotenv_values


DEFAULT_ENV = Path(__file__).resolve().with_name(".env")
INTEGER_FIELDS = {
    "requests_per_day", "avg_input_tokens", "avg_output_tokens",
    "peak_requests_per_minute", "rpm_limit", "tpm_limit",
}


class BudgetResult(TypedDict):
    input_tokens_request: int
    output_tokens_request: int
    total_tokens_request: int
    input_tokens_day: int
    output_tokens_day: int
    total_tokens_day: int
    input_tokens_month: int
    output_tokens_month: int
    total_tokens_month: int
    cost_per_request_usd: float
    daily_cost_usd: float
    monthly_cost_usd: float
    request_cost_with_margin_usd: float
    daily_cost_with_margin_usd: float
    monthly_cost_with_margin_usd: float
    peak_input_tpm: int
    peak_output_tpm: int
    peak_total_tpm: int
    peak_rpm_with_margin: float
    peak_tpm_with_margin: float
    rpm_ok: bool
    tpm_ok: bool
    daily_budget_ok: bool
    monthly_budget_ok: bool
    per_request_budget_ok: bool | None


@dataclass
class FeatureBudget:
    requests_per_day: int
    avg_input_tokens: int
    avg_output_tokens: int
    peak_requests_per_minute: int
    input_price_per_million: float
    output_price_per_million: float
    rpm_limit: int
    tpm_limit: int
    monthly_budget_usd: float
    daily_budget_usd: float
    safety_margin_percent: float
    per_request_budget_usd: float | None = None

    @classmethod
    def from_env(cls, env_path: Path = DEFAULT_ENV) -> "FeatureBudget":
        # Explicit environment variables take precedence; never search parent labs.
        values = {**dotenv_values(env_path, interpolate=False), **os.environ}
        integers: dict[str, int] = {}
        decimals: dict[str, float] = {}
        for field in fields(cls):
            key = field.name.upper()
            raw = values.get(key)
            if field.name == "per_request_budget_usd" and raw is None:
                continue
            if raw is None or not raw.strip():
                raise ValueError(f"{key} is required. Configure it in {env_path}.")
            try:
                if field.name in INTEGER_FIELDS:
                    integers[field.name] = int(raw)
                else:
                    decimals[field.name] = float(raw)
            except ValueError as error:
                unit = "an integer" if field.name in INTEGER_FIELDS else "a number"
                raise ValueError(f"{key} must be {unit}; got {raw!r}.") from error
        return cls(
            requests_per_day=integers["requests_per_day"],
            avg_input_tokens=integers["avg_input_tokens"],
            avg_output_tokens=integers["avg_output_tokens"],
            peak_requests_per_minute=integers["peak_requests_per_minute"],
            input_price_per_million=decimals["input_price_per_million"],
            output_price_per_million=decimals["output_price_per_million"],
            rpm_limit=integers["rpm_limit"],
            tpm_limit=integers["tpm_limit"],
            monthly_budget_usd=decimals["monthly_budget_usd"],
            daily_budget_usd=decimals["daily_budget_usd"],
            safety_margin_percent=decimals["safety_margin_percent"],
            per_request_budget_usd=decimals.get("per_request_budget_usd"),
        )

    def calculate(self) -> BudgetResult:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None and field.name == "per_request_budget_usd":
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field.name.upper()} must be numeric.")
            if field.name in INTEGER_FIELDS and not isinstance(value, int):
                raise ValueError(f"{field.name.upper()} must be an integer.")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field.name.upper()} must be finite and non-negative.")

        input_day = self.requests_per_day * self.avg_input_tokens
        output_day = self.requests_per_day * self.avg_output_tokens
        request_cost = (
            self.avg_input_tokens * self.input_price_per_million
            + self.avg_output_tokens * self.output_price_per_million
        ) / 1_000_000
        daily_cost = request_cost * self.requests_per_day
        monthly_cost = daily_cost * 30
        margin = 1 + self.safety_margin_percent / 100
        peak_input = self.peak_requests_per_minute * self.avg_input_tokens
        peak_output = self.peak_requests_per_minute * self.avg_output_tokens
        peak_rpm = self.peak_requests_per_minute * margin
        peak_tpm = (peak_input + peak_output) * margin

        return {
            "input_tokens_request": self.avg_input_tokens,
            "output_tokens_request": self.avg_output_tokens,
            "total_tokens_request": self.avg_input_tokens + self.avg_output_tokens,
            "input_tokens_day": input_day,
            "output_tokens_day": output_day,
            "total_tokens_day": input_day + output_day,
            "input_tokens_month": input_day * 30,
            "output_tokens_month": output_day * 30,
            "total_tokens_month": (input_day + output_day) * 30,
            "cost_per_request_usd": request_cost,
            "daily_cost_usd": daily_cost,
            "monthly_cost_usd": monthly_cost,
            "request_cost_with_margin_usd": request_cost * margin,
            "daily_cost_with_margin_usd": daily_cost * margin,
            "monthly_cost_with_margin_usd": monthly_cost * margin,
            "peak_input_tpm": peak_input,
            "peak_output_tpm": peak_output,
            "peak_total_tpm": peak_input + peak_output,
            "peak_rpm_with_margin": peak_rpm,
            "peak_tpm_with_margin": peak_tpm,
            "rpm_ok": peak_rpm <= self.rpm_limit,
            "tpm_ok": peak_tpm <= self.tpm_limit,
            "daily_budget_ok": daily_cost * margin <= self.daily_budget_usd,
            "monthly_budget_ok": monthly_cost * margin <= self.monthly_budget_usd,
            "per_request_budget_ok": (
                request_cost * margin <= self.per_request_budget_usd
                if self.per_request_budget_usd is not None else None
            ),
        }


def budget_checks(result: BudgetResult) -> dict[str, bool]:
    checks = {
        "RPM capacity": result["rpm_ok"],
        "TPM capacity": result["tpm_ok"],
        "Daily budget": result["daily_budget_ok"],
        "Monthly budget": result["monthly_budget_ok"],
    }
    if result["per_request_budget_ok"] is not None:
        checks["Per-request budget"] = result["per_request_budget_ok"]
    return checks


def print_report(budget: FeatureBudget, title: str = "TOKEN ECONOMICS REPORT") -> None:
    result = budget.calculate()
    print(f"\n{title}\n{'-' * 60}")
    print(f"Requests/day: {budget.requests_per_day:,}")
    print(f"Input/output tokens per request: "
          f"{budget.avg_input_tokens:,} / {budget.avg_output_tokens:,}")
    print(f"Prices per 1M input/output tokens: "
          f"${budget.input_price_per_million:g} / ${budget.output_price_per_million:g}")
    print(f"Safety margin: {budget.safety_margin_percent:g}% (forecast x "
          f"{1 + budget.safety_margin_percent / 100:g})")
    print(f"Input tokens/day:  {result['input_tokens_day']:,}")
    print(f"Output tokens/day: {result['output_tokens_day']:,}")
    print(f"Total tokens/day:  {result['total_tokens_day']:,}")
    print(f"Input/output tokens/month: {result['input_tokens_month']:,} / "
          f"{result['output_tokens_month']:,}")
    print(f"Total tokens/month (30 days): {result['total_tokens_month']:,}")
    print(f"Estimated cost/request: ${result['cost_per_request_usd']:.6f}")
    print(f"Estimated daily cost:   ${result['daily_cost_usd']:.4f}")
    print(f"Estimated monthly cost: ${result['monthly_cost_usd']:.2f}")
    print(f"Cost/request with margin: ${result['request_cost_with_margin_usd']:.6f}")
    if budget.per_request_budget_usd is not None:
        print(f"Per-request budget: ${budget.per_request_budget_usd:.6f}")
    print(f"Daily cost with margin: ${result['daily_cost_with_margin_usd']:.4f}"
          f" / ${budget.daily_budget_usd:.4f} budget")
    print(f"Monthly cost with margin: ${result['monthly_cost_with_margin_usd']:.2f}"
          f" / ${budget.monthly_budget_usd:.2f} budget")
    print(f"Peak input/output/total TPM (before margin): "
          f"{result['peak_input_tpm']:,} / {result['peak_output_tpm']:,}"
          f" / {result['peak_total_tpm']:,}")
    print(f"Peak RPM with margin: {result['peak_rpm_with_margin']:,.1f}"
          f" / {budget.rpm_limit:,} limit")
    print(f"Peak TPM with margin: {result['peak_tpm_with_margin']:,.0f}"
          f" / {budget.tpm_limit:,} limit")
    checks = budget_checks(result)
    print("\nCHECKS")
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'REVIEW'}")
    if all(checks.values()):
        print("\nRecommendation: GO - WITHIN configured planning limits.")
    else:
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        print(f"\nRecommendation: NO-GO - REVISE before build. Review: {failed}.")
    print("Planning estimate only, not a billing guarantee. Verify live pricing and")
    print("provider quota rules. Total TPM is simplified; retries are not modeled.")


def what_if_scenarios(budget: FeatureBudget) -> list[tuple[str, FeatureBudget]]:
    # First whole-number peak RPM above either capacity threshold.
    margin = 1 + budget.safety_margin_percent / 100
    total_tokens = budget.avg_input_tokens + budget.avg_output_tokens
    capacity = budget.rpm_limit / margin
    if total_tokens:
        capacity = min(capacity, budget.tpm_limit / (total_tokens * margin))
    failing_peak = max(budget.peak_requests_per_minute, math.floor(capacity) + 1)
    return [
        ("Baseline", budget),
        ("More users: requests/day x2",
         replace(budget, requests_per_day=budget.requests_per_day * 2)),
        (f"Traffic spike: peak RPM -> {failing_peak}",
         replace(budget, peak_requests_per_minute=failing_peak)),
        ("Longer context: input tokens/request x2",
         replace(budget, avg_input_tokens=budget.avg_input_tokens * 2)),
        ("Shorter answers: output tokens/request halved",
         replace(budget, avg_output_tokens=budget.avg_output_tokens // 2)),
        ("Smaller prompts: input tokens/request halved",
         replace(budget, avg_input_tokens=budget.avg_input_tokens // 2)),
        ("Higher price: input and output rates x2",
         replace(budget, input_price_per_million=budget.input_price_per_million * 2,
                 output_price_per_million=budget.output_price_per_million * 2)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV,
                        help="Planning .env file (default: beside budget.py).")
    parser.add_argument("--what-if", action="store_true",
                        help="Run baseline and one-change-at-a-time scenarios.")
    args = parser.parse_args()
    if not args.env.is_file():
        parser.error(f"Configuration file not found: {args.env}. Copy .env.example to .env.")
    try:
        budget = FeatureBudget.from_env(args.env)
        budget.calculate()
        if args.what_if:
            for title, scenario in what_if_scenarios(budget):
                print_report(scenario, title)
        else:
            print_report(budget)
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
