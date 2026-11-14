"""Illustrative USD estimates, NOT verified prices or billing totals."""

PRICING = {
    "gemini-3.8-flash": {"input": 0.50, "output": 2.00},
}


def calculate_cost(
    provider: str,
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    if provider == "OllamaProvider":
        return 0.0
    rates = PRICING.get(model)
    if rates is None or input_tokens is None or output_tokens is None:
        return None
    return round(
        (input_tokens * rates["input"] + output_tokens * rates["output"])
        / 1_000_000,
        10,
    )
