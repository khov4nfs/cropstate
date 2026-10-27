"""Explicit FP32 ONNX backend for the SAME multilingual MiniLM-L12 sentence encoder.

This is an execution backend, not a new embedding model. SentenceTransformer
uses a 128-token BERT transformer, attention-masked mean pooling, then L2
normalization. A change of backend can still cause small numeric differences;
record hashes and compare ranking results before treating them as identical.
"""
from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(1 << 20), b""):
            h.update(buf)
    return h.hexdigest()


class ONNXSentenceEncoder:
    """Drop-in encode(texts, normalize_embeddings=True) for HybridRetriever."""

    def __init__(self, model_dir: str | Path | None = None, batch_size: int = 8):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if model_dir is None:
            root = Path.home() / ".cache/huggingface/hub" / (
                "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
            ) / "snapshots"
            candidates = sorted(root.iterdir()) if root.is_dir() else []
            candidates = [p for p in candidates if (p / "onnx/model.onnx").is_file()]
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "Specify --onnx-model-dir with a pinned Hugging Face snapshot "
                    "containing onnx/model.onnx and tokenizer.json."
                )
            model_dir = candidates[0]
        self.model_dir = Path(model_dir).resolve()
        model_path = self.model_dir / "onnx/model.onnx"
        tokenizer_path = self.model_dir / "tokenizer.json"
        if not model_path.is_file() or not tokenizer_path.is_file():
            raise FileNotFoundError("ONNX encoder requires onnx/model.onnx and tokenizer.json")
        self.model_sha256 = sha256_file(model_path)
        self.tokenizer_sha256 = sha256_file(tokenizer_path)
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.tokenizer.enable_truncation(max_length=128)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        if [x.name for x in self.session.get_inputs()] != [
            "input_ids", "attention_mask", "token_type_ids"
        ]:
            raise ValueError("Unexpected ONNX MiniLM inputs: incompatible model export")
        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >=1")
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        encs = self.tokenizer.encode_batch(texts)
        maxlen = max(len(e.ids) for e in encs)
        n = len(encs)
        ids = np.zeros((n, maxlen), dtype=np.int64)
        mask = np.zeros((n, maxlen), dtype=np.int64)
        types = np.zeros((n, maxlen), dtype=np.int64)
        for i, e in enumerate(encs):
            length = len(e.ids)
            ids[i, :length] = e.ids
            mask[i, :length] = e.attention_mask
            types[i, :length] = e.type_ids
        values = self.session.run(["last_hidden_state"], {
            "input_ids": ids, "attention_mask": mask, "token_type_ids": types,
        })[0]
        weights = mask[..., None].astype(np.float32)
        pooled = (values * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        return pooled / np.maximum(norms, 1e-12)

    def encode(self, texts, normalize_embeddings=True):
        if isinstance(texts, str):
            texts = [texts]
        texts = list(texts)
        if not texts:
            return np.empty((0, 384), dtype=np.float32)
        missing = list(dict.fromkeys(x for x in texts if x not in self.cache))
        for start in range(0, len(missing), self.batch_size):
            part = missing[start:start + self.batch_size]
            vectors = self._embed_batch(part)
            for value, vector in zip(part, vectors):
                self.cache[value] = vector
        res = np.stack([self.cache[x] for x in texts])
        if not normalize_embeddings:
            # This adapter intentionally only exposes normalized embeddings,
            # matching CropState's HybridRetriever use; fail rather than lie.
            raise ValueError("ONNXSentenceEncoder only supports normalized embeddings")
        return res
