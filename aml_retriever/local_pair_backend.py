"""Optional CPU cross-encoder; reads an explicitly provisioned local snapshot.

Runtime never downloads weights, executes remote model code or calls an LLM API.
Artifact hashes identify the actual files, rather than trusting a revision label.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class LocalPairBackend:
    def __init__(self):
        path = Path(os.environ.get("AML_PAIR_MODEL_PATH", ""))
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("AML_PAIR_MODEL_PATH must be an absolute snapshot directory")
        self.batch_size = int(os.environ.get("AML_PAIR_BATCH_SIZE", "8"))
        self.max_length = int(os.environ.get("AML_PAIR_MAX_LENGTH", "512"))
        threads = int(os.environ.get("AML_PAIR_THREADS", "1"))
        if self.batch_size < 1 or not 16 <= self.max_length <= 512 or threads < 1:
            raise ValueError("invalid cross-encoder limits")
        files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix in {
            ".json", ".txt", ".safetensors"})
        if not any(p.suffix == ".safetensors" for p in files):
            raise ValueError("cross-encoder requires safetensors weights")
        artifacts = {}
        for artifact in files:
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            artifacts[artifact.name] = digest.hexdigest()
        self.artifact_hashes = artifacts
        self.model_hash = hashlib.sha256(json.dumps(
            [artifacts, self.max_length], sort_keys=True).encode()).hexdigest()
        self.identity = "local-cross-encoder:" + self.model_hash
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        torch.set_num_threads(threads)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False,
            use_safetensors=True).to("cpu").eval()
        if self.model.config.num_labels != 1:
            raise ValueError("cross-encoder must return a single relevance logit")

    def score(self, query, documents):
        # Preserve the full question; truncate document suffixes only. Reject a
        # question that leaves no room for evidence instead of silently cutting it.
        query_length = len(self.tokenizer.encode(query, add_special_tokens=False))
        if query_length + self.tokenizer.num_special_tokens_to_add(pair=True) >= self.max_length:
            raise ValueError("query exceeds cross-encoder context")
        scores = []
        with self.torch.inference_mode():
            for offset in range(0, len(documents), self.batch_size):
                batch = documents[offset:offset + self.batch_size]
                inputs = self.tokenizer([query] * len(batch), list(batch), padding=True,
                                        truncation="only_second", max_length=self.max_length,
                                        return_tensors="pt")
                scores.extend(self.model(**inputs).logits[:, 0].tolist())
        return scores


def create_backend():
    return LocalPairBackend()
