from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_CASES_PATH = Path("eval/cases.json")
DEFAULT_OUTPUT_PATH = Path("eval/reports/latest_report.json")


@dataclass
class CaseResult:
    case_id: str
    domain: str
    status: str
    latency_ms: int | None
    warnings: list[str]
    translated_text: str | None
    response_payload: dict[str, Any] | None
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the translation backend against a fixed list of cases."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the translation API, for example http://127.0.0.1:8000 or an ngrok URL.",
    )
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="Path to the JSON file containing test cases.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the JSON report to write.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout in seconds for each request.",
    )
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("cases file must contain a JSON array")
    return data


def post_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], int]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started_at = time.perf_counter()
    with request.urlopen(http_request, timeout=timeout) as response:
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        response_body = response.read().decode("utf-8")
        return json.loads(response_body), latency_ms


def normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def sentence_count(value: str) -> int:
    count = sum(value.count(symbol) for symbol in ".!?")
    return max(1, count)


def evaluate_case(case: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    source_text = case["text"]
    translated_text = payload.get("translated_text", "")
    normalized_source = normalize_text(source_text)
    normalized_translation = normalize_text(translated_text)

    if not translated_text.strip():
        return ["translated_text is empty"]

    if normalized_translation == normalized_source:
        warnings.append("translation looks unchanged from source text")

    source_length = max(len(source_text.strip()), 1)
    length_ratio = len(translated_text.strip()) / source_length
    min_length_ratio = float(case.get("min_length_ratio", 0.0))
    if min_length_ratio and length_ratio < min_length_ratio:
        warnings.append(
            f"translation is shorter than expected (ratio={length_ratio:.2f}, min={min_length_ratio:.2f})"
        )

    source_sentences = sentence_count(source_text)
    translated_sentences = sentence_count(translated_text)
    if source_sentences >= 2 and translated_sentences < source_sentences:
        warnings.append(
            f"translation may be truncated by sentence count ({translated_sentences} < {source_sentences})"
        )

    lowered_translation = translated_text.lower()
    for index, expected_group in enumerate(case.get("expected_any_of", []), start=1):
        if not any(term.lower() in lowered_translation for term in expected_group):
            warnings.append(
                f"missing expected concept group {index}: one of {', '.join(expected_group)}"
            )

    return warnings


def run_case(base_url: str, case: dict[str, Any], timeout: float) -> CaseResult:
    payload = {
        "text": case["text"],
        "source_lang": case["source_lang"],
        "target_lang": case["target_lang"],
    }
    endpoint = f"{base_url.rstrip('/')}/translate"

    try:
        response_payload, latency_ms = post_json(endpoint, payload, timeout=timeout)
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        return CaseResult(
            case_id=case["id"],
            domain=case.get("domain", "unknown"),
            status="error",
            latency_ms=None,
            warnings=[],
            translated_text=None,
            response_payload=None,
            error=f"HTTP {exc.code}: {response_body}",
        )
    except Exception as exc:
        return CaseResult(
            case_id=case["id"],
            domain=case.get("domain", "unknown"),
            status="error",
            latency_ms=None,
            warnings=[],
            translated_text=None,
            response_payload=None,
            error=str(exc),
        )

    warnings = evaluate_case(case, response_payload)
    status = "warn" if warnings else "ok"
    translated_text = response_payload.get("translated_text")
    return CaseResult(
        case_id=case["id"],
        domain=case.get("domain", "unknown"),
        status=status,
        latency_ms=latency_ms,
        warnings=warnings,
        translated_text=translated_text,
        response_payload=response_payload,
        error=None,
    )


def build_report(
    base_url: str,
    cases_path: Path,
    results: list[CaseResult],
) -> dict[str, Any]:
    total = len(results)
    ok_count = sum(result.status == "ok" for result in results)
    warn_count = sum(result.status == "warn" for result in results)
    error_count = sum(result.status == "error" for result in results)

    average_latency_ms = None
    successful_latencies = [
        result.latency_ms for result in results if result.latency_ms is not None
    ]
    if successful_latencies:
        average_latency_ms = round(sum(successful_latencies) / len(successful_latencies), 2)

    return {
        "base_url": base_url,
        "cases_path": str(cases_path),
        "summary": {
            "total": total,
            "ok": ok_count,
            "warn": warn_count,
            "error": error_count,
            "average_latency_ms": average_latency_ms,
        },
        "results": [
            {
                "case_id": result.case_id,
                "domain": result.domain,
                "status": result.status,
                "latency_ms": result.latency_ms,
                "warnings": result.warnings,
                "translated_text": result.translated_text,
                "error": result.error,
                "response_payload": result.response_payload,
            }
            for result in results
        ],
    }


def print_result(result: CaseResult) -> None:
    latency = f"{result.latency_ms}ms" if result.latency_ms is not None else "-"
    print(f"[{result.status.upper():5}] {result.case_id} ({result.domain}) {latency}")
    if result.error:
        print(f"  error: {result.error}")
        return

    assert result.translated_text is not None
    print(f"  translation: {result.translated_text}")
    for warning in result.warnings:
        print(f"  warning: {warning}")


def main() -> int:
    args = parse_args()
    cases_path = Path(args.cases)
    output_path = Path(args.output)
    cases = load_cases(cases_path)

    results: list[CaseResult] = []
    for case in cases:
        result = run_case(args.base_url, case, timeout=args.timeout)
        results.append(result)
        print_result(result)

    report = build_report(args.base_url, cases_path, results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = report["summary"]
    print("")
    print(
        "Summary:"
        f" total={summary['total']}"
        f" ok={summary['ok']}"
        f" warn={summary['warn']}"
        f" error={summary['error']}"
        f" avg_latency_ms={summary['average_latency_ms']}"
    )
    print(f"Report written to {output_path}")

    return 1 if summary["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
