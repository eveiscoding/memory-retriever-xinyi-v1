"""Standard-library AML Add/Search client with contract validation."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


class AMLClient:
    def __init__(self, base_url: str, *, api_key: str = "", timeout: float = 1800.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _post(self, path: str, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {error.code} {path}: {body[:500]}") from error
        value = json.loads(body)
        if not isinstance(value, dict):
            raise RuntimeError(f"{path}: response must be a JSON object")
        return value

    def add(self, payload: dict) -> dict:
        response = self._post("/add", payload)
        if response.get("success") is not True:
            raise RuntimeError("/add: success must be true")
        for key in ("request_id", "user_id", "session_id"):
            if response.get(key) != payload.get(key):
                raise RuntimeError(f"/add: {key} was not echoed exactly")
        return response

    def search(self, payload: dict) -> list[dict]:
        response = self._post("/search", payload)
        data = response.get("data")
        if not isinstance(data, list):
            raise RuntimeError("/search: data must be an array")
        if len(data) > payload["top_k"]:
            raise RuntimeError("/search: result count exceeds top_k")
        for item in data:
            if not isinstance(item, dict) or not item.get("id") or not item.get("content"):
                raise RuntimeError("/search: every item needs id and content")
        return data
