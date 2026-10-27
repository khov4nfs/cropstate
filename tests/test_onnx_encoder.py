"""Regression tests for numerical ONNX adapter pooling; no pretrained weights needed."""
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np

from cropstate.onnx_encoder import ONNXSentenceEncoder, sha256_file


class TokenRow:
    def __init__(self, ids):
        self.ids = ids
        self.type_ids = [0] * len(ids)
        self.attention_mask = [1] * len(ids)


class FakeTokenizer:
    def encode_batch(self, texts):
        return [TokenRow([len(s), 2]) for s in texts]


class FakeSession:
    def run(self, outputs, inputs):
        assert outputs == ["last_hidden_state"]
        ids = inputs["input_ids"]
        n, t = ids.shape
        embeddings = np.zeros((n, t, 384), dtype=np.float32)
        embeddings[:, :, 0] = ids.astype(np.float32)
        embeddings[:, :, 1] = 1.0
        return [embeddings]


class ONNXAdapterTests(unittest.TestCase):
    def make_encoder(self):
        obj = ONNXSentenceEncoder.__new__(ONNXSentenceEncoder)
        obj.tokenizer = FakeTokenizer()
        obj.session = FakeSession()
        obj.batch_size = 2
        obj.cache = OrderedDict()
        return obj

    def test_masked_mean_normalized_and_cached(self):
        encoder = self.make_encoder()
        embeddings = encoder.encode(["crop", "rice", "crop"])
        self.assertEqual(embeddings.shape, (3, 384))
        np.testing.assert_allclose(np.linalg.norm(embeddings, axis=1), 1.0)
        np.testing.assert_allclose(embeddings[0], embeddings[2])
        self.assertEqual(set(encoder.cache), {"crop", "rice"})
        np.testing.assert_allclose(encoder.encode(["rice"]), embeddings[1:2])

    def test_empty_input_and_rejected_raw_embeddings(self):
        encoder = self.make_encoder()
        self.assertEqual(encoder.encode([]).shape, (0, 384))
        with self.assertRaisesRegex(ValueError, "normalized embeddings"):
            encoder.encode(["rice"], normalize_embeddings=False)

    def test_file_digest(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.txt"
            p.write_bytes(b"ONNX-provenance")
            self.assertEqual(sha256_file(p), hashlib.sha256(b"ONNX-provenance").hexdigest())


if __name__ == "__main__":
    unittest.main()
