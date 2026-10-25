from __future__ import annotations

import argparse
from pathlib import Path

from estimator import HemoglobinEstimator


def _print_result(result: dict) -> None:
    print()
    if result.get("status") == "invalid_image":
        print("Result: INVALID IMAGE")
        print(f"Reason: {result.get('message')}")
        for name, info in (result.get("validation") or {}).items():
            print(f"  {name}: valid={info['valid']} score={info['score']:.2f} - {info['message']}")
        return

    print("Result: SUCCESS")
    print(f"  Hemoglobin: {result['hemoglobin_estimate']} {result['unit']}")
    print(f"  Status: {result['health_status']} - {result['health_message']}")
    print(f"  Model: {result.get('model')}")
    if result.get("modalities_used"):
        print(f"  Modalities: {', '.join(result['modalities_used'])}")
    if result.get("eye_quality_score") is not None:
        print(f"  Eye quality: {result['eye_quality_score']:.2f}")
    if result.get("warning"):
        print(f"  Note: {result['warning']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Local HemoLens hemoglobin estimator")
    parser.add_argument("--eye", type=Path, help="Path to an eye image")
    parser.add_argument("--nail", type=Path, help="Path to a nail image")
    parser.add_argument("--palm", type=Path, help="Path to a palm image")
    parser.add_argument(
        "--eye-only",
        action="store_true",
        help="Prefer the eye-only ridge model when only an eye image is supplied",
    )
    args = parser.parse_args()

    if not any([args.eye, args.nail, args.palm]):
        parser.error("Provide at least one of --eye, --nail, or --palm")

    estimator = HemoglobinEstimator()
    print("Models loaded:")
    print(f"  Eye model: {'yes' if estimator.status()['eye_model_loaded'] else 'no'}")
    print(f"  Multimodal model: {'yes' if estimator.status()['multimodal_model_loaded'] else 'no'}")

    result = estimator.predict_from_paths(
        eye_path=args.eye,
        nail_path=args.nail,
        palm_path=args.palm,
        prefer_multimodal=not args.eye_only,
    )
    _print_result(result)
    return 0 if result.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
