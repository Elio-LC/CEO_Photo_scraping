"""
DashScope Vision API integration for validating extracted CEO photos.
"""

from __future__ import annotations

import base64
import json
import http.client
import concurrent.futures
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class _RateLimiter:
    """Simple token bucket to keep QPS within limits."""

    def __init__(self, max_calls: int, interval_seconds: float = 1.0) -> None:
        self.max_calls = max(1, max_calls)
        self.interval = max(interval_seconds, 0.001)
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def wait_for_slot(self) -> None:
        while True:
            with self._lock:
                now = time.time()
                while self._timestamps and now - self._timestamps[0] >= self.interval:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_calls:
                    self._timestamps.append(now)
                    return
                oldest = self._timestamps[0]
                sleep_time = max(self.interval - (now - oldest), 0.001)
            time.sleep(sleep_time)


class DashscopeVisionClient:
    """DashScope Vision API client supporting sync or async modes."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        concise_mode: bool = False,
        async_mode: bool = False,
        max_qps: int = 15,
        poll_interval: float = 2.0,
        async_timeout: float = 60.0,
    ) -> None:
        self.api_key = api_key or self._load_api_key()
        self.ready = bool(self.api_key)
        self.concise_mode = concise_mode
        self.async_mode = async_mode
        self.poll_interval = poll_interval
        self.async_timeout = async_timeout
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
        self._rate_limiter = _RateLimiter(max_qps if max_qps else 15) if async_mode else None

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _load_api_key(self) -> Optional[str]:
        from pathlib import Path

        try:
            key_file = Path(__file__).parent / "dashscope_api_key.txt"
            if key_file.exists():
                return key_file.read_text().strip()
        except Exception:
            pass
        return None

    def is_ready(self) -> bool:
        return self.ready

    def _prompt_text(self) -> str:
        if self.concise_mode:
            return (
                "Decide only whether this image is an acceptable CEO portrait. "
                "Return minimal JSON and nothing else: "
                '{"is_valid": true/false, "confidence": 0.0-1.0}'
            )
        return """Analyze this image and decide if it is a valid CEO portrait.

Requirements:
1. Face must be clearly visible (not a back view or hidden).
2. No mask covering the face (sunglasses are fine).
3. Exclude charts, graphics, logos, and non-human images.
4. Photo quality must be clear.
5. Group photos are acceptable if faces are visible and mask-free.

Respond only with JSON in this exact format and no extra text:
{
    "is_valid": true/false,
    "confidence": 0.0-1.0,
    "reason": "short reason"
}"""

    def _build_payload(self, image_base64: str) -> Dict[str, Any]:
        return {
            "model": "qwen-vl-max",
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"image": f"data:image/jpeg;base64,{image_base64}"},
                            {"text": self._prompt_text()},
                        ],
                    }
                ]
            },
            "parameters": {"result_format": "message"},
        }

    def _build_face_compare_payload(
        self, image_base64_list: List[str], labels: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        content: List[Dict[str, str]] = []
        for encoded in image_base64_list:
            content.append({"image": f"data:image/jpeg;base64,{encoded}"})
        label_text = ""
        if labels:
            parts = [f"{idx+1}:{label}" for idx, label in enumerate(labels)]
            label_text = "Image-to-index mapping: " + "; ".join(parts) + ". "
        prompt = (
            "These are photos of the same company's CEO across different years, ordered by index (1..N). "
            + label_text
            + "Identify the face that appears most often (assume that is the real CEO) and return minimal JSON: "
            '{"verdict":"same|mixed|multi|uncertain","confidence":0.0-1.0,"majority":[1,2],"outliers":[3],"multi":[4]} '
            "majority: indices that match the true CEO (flips/mirrors/minor pose changes still count as same person; "
            "if glasses are present in only some images, treat as different faces). "
            "If gender or hairstyle obviously differ, treat as different people. "
            "outliers: indices that are not the CEO and should be removed. "
            "multi: indices in group photos where the CEO cannot be uniquely identified (do not place these in majority/outliers). "
            "If all images match -> verdict=same and outliers is empty. "
            "If some differ -> verdict=mixed and list outliers. "
            "If group photos prevent a decision -> verdict=multi. "
            "If uncertain -> verdict=uncertain. "
            "Do not return anything except the JSON."
        )
        content.append({"text": prompt})
        return {
            "model": "qwen-vl-max",
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": {"result_format": "message"},
        }

    def _http_post(self, path: str, payload: str) -> Dict[str, Any]:
        conn = http.client.HTTPSConnection("dashscope.aliyuncs.com", timeout=90)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        conn.request("POST", path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return json.loads(data.decode("utf-8"))

    def _http_get(self, path: str) -> Dict[str, Any]:
        conn = http.client.HTTPSConnection("dashscope.aliyuncs.com", timeout=60)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        conn.request("GET", path, headers=headers)
        res = conn.getresponse()
        data = res.read()
        conn.close()
        return json.loads(data.decode("utf-8"))

    def _normalize_content(self, content: Any) -> Optional[str]:
        if content is None:
            return None
        if isinstance(content, str):
            stripped = content.strip()
            return stripped or None
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text = item["text"].strip()
                    if text:
                        parts.append(text)
            text = "\n".join(parts).strip()
            return text or None
        if isinstance(content, dict):
            return self._normalize_content(content.get("content") or content.get("text"))
        return None

    def _extract_text_from_payload(self, payload: Dict[str, Any]) -> Optional[str]:
        if not payload:
            return None
        if "choices" in payload and payload["choices"]:
            message = payload["choices"][0].get("message", {})
            return self._normalize_content(message.get("content"))
        if "result" in payload and isinstance(payload["result"], dict):
            return self._extract_text_from_payload(payload["result"])
        if "results" in payload and payload["results"]:
            first = payload["results"][0]
            if isinstance(first, dict):
                if "text" in first:
                    return self._normalize_content(first["text"])
                if "content" in first:
                    return self._normalize_content(first["content"])
        if "message" in payload and isinstance(payload["message"], dict):
            return self._normalize_content(payload["message"].get("content"))
        if "content" in payload:
            return self._normalize_content(payload["content"])
        return None

    def _parse_json_block(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        candidate = text.strip()
        if "```json" in candidate:
            candidate = candidate.split("```json", 1)[1]
            candidate = candidate.split("```", 1)[0]
        elif "```" in candidate:
            candidate = candidate.split("```", 1)[1]
            candidate = candidate.split("```", 1)[0]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    def _result_from_json(self, data: Dict[str, Any]) -> Dict[str, Any]:
        reason_value = data.get("reason")
        if reason_value is None and self.concise_mode:
            reason_value = "concise_mode"
        return {
            "is_person": bool(data.get("is_valid", False)),
            "confidence": float(data.get("confidence", 0.0)),
            "reason": reason_value if reason_value is not None else "Unknown",
        }

    def _parse_result_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        text = self._extract_text_from_payload(payload)
        if text:
            parsed = self._parse_json_block(text)
            if parsed:
                return self._result_from_json(parsed)
        status = payload.get("task_status")
        message = payload.get("message") or payload.get("reason") or payload.get("code") or "dashscope_unknown_response"
        if status and status not in {"SUCCEEDED", "success"}:
            message = f"{message}|status={status}"
        return {
            "is_person": False,
            "confidence": 0.0,
            "reason": message,
        }

    def _parse_face_compare_output(
        self, payload: Dict[str, Any]
    ) -> Tuple[str, float, List[int], List[int], List[int]]:
        text = self._extract_text_from_payload(payload)
        if text:
            parsed = self._parse_json_block(text)
            if parsed:
                verdict = str(parsed.get("verdict", "uncertain")).lower()
                if verdict not in {"same", "mixed", "uncertain", "multi"}:
                    verdict = "uncertain"
                confidence = float(parsed.get("confidence", 0.0))
                majority_raw = parsed.get("majority")
                outliers_raw = parsed.get("outliers")
                multi_raw = parsed.get("multi")

                def _normalize_indices(value: Any) -> List[int]:
                    if not isinstance(value, list):
                        return []
                    cleaned: List[int] = []
                    for item in value:
                        try:
                            idx = int(item)
                        except (TypeError, ValueError):
                            continue
                        if idx >= 1:
                            cleaned.append(idx)
                    return cleaned

                return (
                    verdict,
                    confidence,
                    _normalize_indices(majority_raw),
                    _normalize_indices(outliers_raw),
                    _normalize_indices(multi_raw),
                )
        message = payload.get("message") or payload.get("reason") or "face_compare_failed"
        return "uncertain", 0.0, [], [], []

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _call_api_sync(self, image_bytes: bytes) -> Dict[str, Any]:
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = self._build_payload(image_base64)
            response = self._http_post(
                "/api/v1/services/aigc/multimodal-generation/generation",
                json.dumps(payload),
            )
            output = response.get("output")
            if not output:
                return {
                    "is_person": False,
                    "confidence": 0.0,
                    "reason": response.get("message", "missing_output"),
                }
            return self._parse_result_payload(output)
        except Exception as exc:  # noqa: BLE001
            return {
                "is_person": False,
                "confidence": 0.0,
                "reason": f"Error: {exc}",
            }

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _limit_async_call(self) -> None:
        if self._rate_limiter is not None:
            self._rate_limiter.wait_for_slot()

    def _submit_async_task(self, image_bytes: bytes) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        try:
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            payload = self._build_payload(image_base64)
            self._limit_async_call()
            response = self._http_post("/api/v1/tasks", json.dumps(payload))
            output = response.get("output", {})
            task_id = output.get("task_id") or response.get("task_id")
            if task_id:
                return task_id, None
            return None, {
                "is_person": False,
                "confidence": 0.0,
                "reason": output.get("message", "async_no_task_id"),
            }
        except Exception as exc:  # noqa: BLE001
            return None, {
                "is_person": False,
                "confidence": 0.0,
                "reason": f"async_submit_error:{exc}",
            }

    def _fetch_async_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        try:
            self._limit_async_call()
            return self._http_get(f"/api/v1/tasks/{task_id}")
        except Exception:
            return None

    def _parse_async_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        output = response.get("output", {})
        task_status = output.get("task_status") or output.get("status")
        if task_status not in {"SUCCEEDED", "success", None}:
            message = output.get("message") or f"async_status_{task_status}"
            return {
                "is_person": False,
                "confidence": 0.0,
                "reason": message,
            }
        payload = output.get("result") or output
        return self._parse_result_payload(payload)

    def _analyze_portraits_batch_async(
        self,
        images_data: List[Tuple[bytes, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        if not images_data:
            return []

        results: List[Optional[Dict[str, Any]]] = [None] * len(images_data)
        pending: Dict[str, int] = {}

        for idx, (image_bytes, _person_name) in enumerate(images_data):
            task_id, immediate = self._submit_async_task(image_bytes)
            if task_id:
                pending[task_id] = idx
            else:
                results[idx] = immediate

        if not pending:
            return [
                res
                or {
                    "is_person": False,
                    "confidence": 0.0,
                    "reason": "async_submission_failed",
                }
                for res in results
            ]

        start_time = time.time()
        while pending:
            completed: List[Tuple[str, int, Dict[str, Any]]] = []
            for task_id, idx in list(pending.items()):
                response = self._fetch_async_task(task_id)
                if not response:
                    continue
                output = response.get("output", {})
                status = output.get("task_status") or output.get("status")
                if status in {"SUCCEEDED", "success", "FAILED", "CANCELED", "UNKNOWN"} or output.get("result"):
                    completed.append((task_id, idx, response))
            for task_id, idx, response in completed:
                pending.pop(task_id, None)
                results[idx] = self._parse_async_response(response)
            if pending:
                if time.time() - start_time > self.async_timeout:
                    for idx in pending.values():
                        results[idx] = {
                            "is_person": False,
                            "confidence": 0.0,
                            "reason": "async_timeout",
                        }
                    pending.clear()
                else:
                    time.sleep(self.poll_interval)

        return [
            res
            or {
                "is_person": False,
                "confidence": 0.0,
                "reason": "async_unknown_error",
            }
            for res in results
        ]

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def analyze_portrait(self, image_bytes: bytes, person_name: Optional[str] = None) -> Dict[str, Any]:
        if not self.is_ready():
            return {
                "is_person": False,
                "confidence": 0.0,
                "reason": "API key not configured",
            }
        if self.async_mode:
            return self._analyze_portraits_batch_async([(image_bytes, person_name)])[0]
        return self._call_api_sync(image_bytes)

    def analyze_portraits_batch(
        self,
        images_data: List[Tuple[bytes, Optional[str]]],
    ) -> List[Dict[str, Any]]:
        if not self.is_ready():
            return [
                {
                    "is_person": False,
                    "confidence": 0.0,
                    "reason": "API key not configured",
                }
                for _ in images_data
            ]

        if self.async_mode:
            return self._analyze_portraits_batch_async(images_data)

        if len(images_data) < 5:
            futures = [
                self.executor.submit(self._call_api_sync, image_bytes)
                for image_bytes, _person_name in images_data
            ]
            ordered = [None] * len(images_data)
            for idx, future in enumerate(futures):
                ordered[idx] = future.result()
            return ordered
        return [self._call_api_sync(image_bytes) for image_bytes, _ in images_data]

    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    def _encode_image_file(self, image_path: Path) -> Optional[str]:
        try:
            data = image_path.read_bytes()
        except OSError:
            return None
        return base64.b64encode(data).decode("utf-8")

    def compare_faces_detailed(
        self,
        image_paths: List[str],
        labels: Optional[List[str]] = None,
    ) -> Tuple[str, float, List[int], List[int], List[int]]:
        if len(image_paths) < 2:
            return "insufficient", 0.0, [], [], []
        if not self.is_ready():
            return "uncertain", 0.0, [], [], []
        encoded_images: List[str] = []
        missing_files: List[str] = []
        for path_str in image_paths:
            path = Path(path_str)
            encoded = self._encode_image_file(path)
            if encoded is None:
                missing_files.append(str(path))
            else:
                encoded_images.append(encoded)
        if len(encoded_images) < 2:
            return "insufficient", 0.0, [], [], []
        payload = self._build_face_compare_payload(encoded_images, labels)
        try:
            response = self._http_post(
                "/api/v1/services/aigc/multimodal-generation/generation",
                json.dumps(payload),
            )
            output = response.get("output") or {}
            return self._parse_face_compare_output(output)
        except Exception as exc:  # noqa: BLE001
            return "uncertain", 0.0, [], [], []


def compare_faces(
    image_paths: List[str],
    client: Optional[DashscopeVisionClient] = None,
    labels: Optional[List[str]] = None,
) -> Tuple[str, float, List[int], List[int], List[int]]:
    """Convenience wrapper used by verification scripts."""
    local_client = client or DashscopeVisionClient()
    return local_client.compare_faces_detailed(image_paths, labels)
