"""Health check script for DashScope Vision API."""

from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path
from typing import Optional

from dashscope_vision import DashscopeVisionClient

_FALLBACK_IMAGE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGMAAQAABQABDQottAAAAABJRU5ErkJggg=="


def _load_image_bytes(custom_path: Optional[str]) -> bytes:
    if custom_path:
        image_path = Path(custom_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Custom image not found: {image_path}")
        return image_path.read_bytes()
    return base64.b64decode(_FALLBACK_IMAGE_BASE64)


def main() -> int:
    parser = argparse.ArgumentParser(description="Call DashScope Vision API for a basic health check")
    parser.add_argument("--image", help="Optional: path to a test image (defaults to built-in 1x1 PNG)")
    parser.add_argument(
        "--expect-person",
        action="store_true",
        help="Require is_person=True for success if set",
    )
    args = parser.parse_args()

    client = DashscopeVisionClient()
    if not client.is_ready():
        print("ERROR: DashScope API key not found (dashscope_api_key.txt)")
        return 2

    try:
        image_bytes = _load_image_bytes(args.image)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: Unable to read test image: {exc}")
        return 3

    result = client.analyze_portrait(image_bytes, person_name="Health Check")
    reason = result.get("reason", "")

    if reason.lower().startswith("error") or reason.lower().startswith("api error"):
        print(f"ERROR: API call failed: {reason}")
        return 4

    is_person = bool(result.get("is_person"))
    confidence = float(result.get("confidence", 0.0))

    if args.expect_person and not is_person:
        print(
            f"WARNING: API returned non-person (confidence={confidence:.2f}, reason={reason}); API reachable."
        )
        return 5

    print(
        "OK: DashScope call succeeded | is_person={} | confidence={:.2f} | reason={}".format(
            is_person,
            confidence,
            reason,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

# python check_dashscope_health.py --image "D:\ceo_photo_pool\0000001800_2023.png"
