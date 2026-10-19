"""Optional local ONNX cross-encoder; no downloads or remote model code.

The checkpoint/quantization are explicit artifacts, not assumed equivalent to an
original precision model. Every artifact's bytes contribute to its identity.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


class OnnxPairBackend:
    def __init__(self):
        path = Path(os.environ.get("AML_PAIR_MODEL_PATH", ""))
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("AML_PAIR_MODEL_PATH must be an absolute snapshot directory")
        self.batch_size = int(os.environ.get("AML_PAIR_BATCH_SIZE", "1"))
        self.max_length = int(os.environ.get("AML_PAIR_MAX_LENGTH", "512"))
        threads = int(os.environ.get("AML_PAIR_THREADS", "1"))
        if self.batch_size < 1 or not 16 <= self.max_length <= 8192 or threads < 1:
            raise ValueError("invalid ONNX cross-encoder limits")
        filename = os.environ.get("AML_PAIR_ONNX_FILE", "onnx/model_int8.onnx")
        model_path = (path / filename).resolve()
        if path.resolve() not in model_path.parents or not model_path.is_file():
            raise ValueError("ONNX model must exist inside the explicit snapshot")
        # External tensor files are not supported by this adapter. Record all
        # tokenizer/model assets but exclude Hugging Face download bookkeeping.
        files = sorted(p for p in path.rglob("*") if p.is_file() and
                       ".cache" not in p.relative_to(path).parts)
        artifacts = {}
        for artifact in files:
            digest = hashlib.sha256()
            with artifact.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            artifacts[artifact.relative_to(path).as_posix()] = digest.hexdigest()
        self.artifact_hashes = artifacts
        self.model_hash = hashlib.sha256(json.dumps(
            [artifacts, filename, self.max_length], sort_keys=True).encode()).hexdigest()
        self.identity = "onnx-cross-encoder:" + self.model_hash
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(path), local_files_only=True, trust_remote_code=False)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]

    def score(self, query, documents):
        import numpy as np
        length = len(self.tokenizer.encode(query, add_special_tokens=False))
        if length + self.tokenizer.num_special_tokens_to_add(pair=True) >= self.max_length:
            raise ValueError("query exceeds ONNX context")
        scores = []
        for start in range(0, len(documents), self.batch_size):
            batch = documents[start:start + self.batch_size]
            inputs = self.tokenizer([query] * len(batch), list(batch), padding=True,
                                    truncation="only_second", max_length=self.max_length,
                                    return_tensors="np")
            logits = np.asarray(self.session.run(
                None, {key: inputs[key].astype(np.int64, copy=False)
                       for key in self.input_names})[0])
            if logits.shape != (len(batch), 1) or not np.isfinite(logits).all():
                raise ValueError("ONNX cross-encoder must return one finite logit per pair")
            scores.extend(logits[:, 0].tolist())
        return scores


def create_backend():
    return OnnxPairBackend()
