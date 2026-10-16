import io
import json
import os
import unittest
import urllib.error
from unittest.mock import patch

from aml_retriever.dashscope_embedding import DashScopeEmbeddingBackend
from aml_retriever.embeddings import load_embedding_backend


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _response(start: int, count: int, dimension: int) -> _Response:
    data = [
        {"index": index, "embedding": [float(start + index)] * dimension}
        for index in reversed(range(count))
    ]
    return _Response(json.dumps({"data": data}).encode("utf-8"))


class TestDashScopeEmbeddingBackend(unittest.TestCase):
    def test_requires_credentials_and_workspace_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "DASHSCOPE_API_KEY"):
                DashScopeEmbeddingBackend(dimension=64)

    def test_loader_batches_requests_and_preserves_input_order(self):
        env = {
            "DASHSCOPE_API_KEY": "test-secret",
            "DASHSCOPE_BASE_URL": "https://workspace.example/compatible-mode/v1",
        }
        with patch.dict(os.environ, env, clear=True), patch(
            "urllib.request.urlopen",
            side_effect=[_response(0, 10, 64), _response(10, 1, 64)],
        ) as urlopen:
            backend = load_embedding_backend(
                "aml_retriever.dashscope_embedding:create_backend", dimension=64
            )
            vectors = backend.embed([f"text-{index}" for index in range(11)])

        self.assertEqual(len(vectors), 11)
        self.assertEqual(vectors[0], [0.0] * 64)
        self.assertEqual(vectors[10], [10.0] * 64)
        self.assertEqual(urlopen.call_count, 2)
        first_request = urlopen.call_args_list[0].args[0]
        first_payload = json.loads(first_request.data)
        self.assertEqual(first_request.full_url, env["DASHSCOPE_BASE_URL"] + "/embeddings")
        self.assertEqual(first_request.headers["Authorization"], "Bearer test-secret")
        self.assertEqual(first_payload["model"], "text-embedding-v4")
        self.assertEqual(first_payload["dimensions"], 64)
        self.assertEqual(len(first_payload["input"]), 10)

    def test_retries_rate_limit(self):
        error = urllib.error.HTTPError(
            "https://workspace.example/embeddings",
            429,
            "rate limited",
            {},
            io.BytesIO(b'{"error":{"code":"rate_limit","message":"slow down"}}'),
        )
        backend = DashScopeEmbeddingBackend(
            dimension=64,
            api_key="test-secret",
            base_url="https://workspace.example/compatible-mode/v1",
            max_retries=1,
        )
        with patch(
            "urllib.request.urlopen", side_effect=[error, _response(0, 1, 64)]
        ) as urlopen, patch("time.sleep"):
            vectors = backend.embed(["hello"])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(urlopen.call_count, 2)

    def test_rejects_wrong_vector_dimension(self):
        backend = DashScopeEmbeddingBackend(
            dimension=64,
            api_key="test-secret",
            base_url="https://workspace.example/compatible-mode/v1",
        )
        with patch("urllib.request.urlopen", return_value=_response(0, 1, 63)):
            with self.assertRaisesRegex(RuntimeError, "embedding dimension"):
                backend.embed(["hello"])


if __name__ == "__main__":
    unittest.main()
