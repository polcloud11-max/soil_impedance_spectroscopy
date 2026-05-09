"""
Continuous onboard processing loop for the rover impedance probe.

Each iteration:

    1. Read one sensor sweep payload (JSON, NDJSON line, or array file)
    2. Convert it into a :class:`Spectrum`
    3. Fit the ZARC equivalent-circuit model
    4. Map the fit parameters to estimated VWC and bulk capacitance
    5. Emit one clean JSON object on stdout following the rover schema

No plotting, no CSV I/O, no machine-learning models in this path.

Usage
-----
Stream mode (default) - one sweep per stdin line, emit one JSON per sweep::

    $ python main.py < live_bus.ndjson

Batch mode - read a JSON array from a file::

    $ python main.py --input sweeps.json

In both modes ``--log-file <path>`` mirrors stdout to a file.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, IO, Optional

from src.data_loading import load_sensor_sweep
from src.moisture_calibration import calculate_vwc
from src.preprocessing import Spectrum
from src.utils import get_logger
from src.zarc_fitting import ZarcFit, fit_spectrum

log = get_logger()


# ---------------------------------------------------------------------
# Output schema construction
# ---------------------------------------------------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_output_record(
    spectrum: Spectrum,
    fit: ZarcFit,
    soil_metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Assemble the JSON payload required by the rover spec."""
    timestamp = spectrum.timestamp or _utc_now_iso()
    return {
        "timestamp": timestamp,
        "sample_id": spectrum.sample_id,
        "zarc_parameters": {
            "Rs": fit.Rs,
            "R": fit.R,
            "tau": fit.tau,
            "alpha": fit.alpha,
        },
        "soil_metrics": {
            "vwc_percent": soil_metrics["vwc_percent"],
            "bulk_capacitance_farads": soil_metrics["bulk_capacitance_farads"],
        },
        "fit_health": {
            "success": fit.success,
            "rmse_rel": fit.rmse_rel,
            "n_points": fit.n_points,
            "message": fit.message,
        },
    }


# ---------------------------------------------------------------------
# Single-sweep processing
# ---------------------------------------------------------------------
def process_sweep(payload: Any) -> Dict[str, Any]:
    """Run the full pipeline on one sweep and return the output record."""
    spectrum = load_sensor_sweep(payload)
    fit = fit_spectrum(spectrum)
    soil_metrics = calculate_vwc(fit.Rs, fit.R, fit.tau, fit.alpha)
    return build_output_record(spectrum, fit, soil_metrics)


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------
def _emit(record: Dict[str, Any], sinks: Iterable[IO[str]]) -> None:
    line = json.dumps(record, allow_nan=True, separators=(",", ":"))
    for sink in sinks:
        sink.write(line + "\n")
        sink.flush()


def _iter_stdin_payloads(stream: IO[str]) -> Iterable[Any]:
    """Yield one parsed JSON payload per non-blank line on stdin."""
    for raw in stream:
        raw = raw.strip()
        if not raw:
            continue
        try:
            yield json.loads(raw)
        except json.JSONDecodeError as exc:
            log.error(f"Malformed JSON on stdin: {exc}")
            continue


def _iter_file_payloads(path: Path) -> Iterable[Any]:
    """Yield payloads from a JSON file containing either an array or NDJSON."""
    text = path.read_text()
    text_stripped = text.lstrip()
    if text_stripped.startswith("["):
        # JSON array of payloads
        try:
            payloads = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"Invalid JSON array in {path}: {exc}") from exc
        if not isinstance(payloads, list):
            raise SystemExit(f"Expected an array in {path}")
        for p in payloads:
            yield p
    else:
        # NDJSON: one payload per line
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                log.error(f"Malformed NDJSON line: {exc}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Onboard impedance-spectroscopy processing loop."
    )
    parser.add_argument(
        "--input", "-i", type=Path, default=None,
        help=("Path to a JSON file (array of payloads) or NDJSON file. "
              "If omitted, payloads are read from stdin (one per line)."),
    )
    parser.add_argument(
        "--log-file", "-l", type=Path, default=None,
        help="If given, also write each output record (NDJSON) to this file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)

    sinks = [sys.stdout]
    log_handle: Optional[IO[str]] = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = args.log_file.open("a", encoding="utf-8")
        sinks.append(log_handle)

    if args.input is not None:
        payload_iter = _iter_file_payloads(args.input)
    else:
        payload_iter = _iter_stdin_payloads(sys.stdin)

    n_ok = 0
    n_err = 0
    try:
        for payload in payload_iter:
            try:
                record = process_sweep(payload)
                _emit(record, sinks)
                n_ok += 1
            except ValueError as exc:
                # Bad payload (validation error) -> emit a structured error
                err_record = {
                    "timestamp": _utc_now_iso(),
                    "sample_id": (payload.get("sample_id")
                                  if isinstance(payload, dict) else None),
                    "error": str(exc),
                    "fit_health": {"success": False, "rmse_rel": None,
                                   "n_points": 0, "message": str(exc)},
                }
                _emit(err_record, sinks)
                n_err += 1
    finally:
        if log_handle is not None:
            log_handle.close()

    log.info(f"processed sweeps: ok={n_ok} err={n_err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
