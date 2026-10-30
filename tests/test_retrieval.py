import unittest

import numpy as np

from cropstate.knowledge import KnowledgeChunk
from cropstate.retrieval import rerank, rerank_scores


class RetrievalTest(unittest.TestCase):
    def test_vector_changes_order(self):
        first = KnowledgeChunk(
            chunk_id="first",
            text="This complete test record contains enough words to validate a deterministic ranking path without loading any external model or remote resource during the unit test.",
            topic="topic_a",
            stage_compatibility=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            authority_score=0.5,
        )
        second = KnowledgeChunk(
            chunk_id="second",
            text="This second complete record also contains enough words to exercise deterministic ranking while remaining independent from external files services models and network access.",
            topic="topic_a",
            stage_compatibility=(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            authority_score=0.5,
        )
        ranked = rerank(
            {"first": 0.5, "second": 0.5},
            {"first": first, "second": second},
            np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            1.0,
        )
        self.assertEqual(ranked[0], "first")

    def test_authority_confound_can_be_disabled(self):
        a = KnowledgeChunk(
            chunk_id="a", text="test a", topic="x",
            stage_compatibility=(1, 0, 0, 0, 0, 0), authority_score=0.0,
        )
        b = KnowledgeChunk(
            chunk_id="b", text="test b", topic="x",
            stage_compatibility=(1, 0, 0, 0, 0, 0), authority_score=1.0,
        )
        base = {"a": 0.5, "b": 0.5}
        chunks = {"a": a, "b": b}
        belief = np.array([1, 0, 0, 0, 0, 0], dtype=float)
        fair = rerank_scores(base, chunks, belief, 1.0, source_weight=0.0)
        self.assertAlmostEqual(fair["a"]["final_score"], fair["b"]["final_score"])
        weighted = rerank_scores(base, chunks, belief, 1.0, source_weight=0.1)
        self.assertGreater(weighted["b"]["final_score"], weighted["a"]["final_score"])


if __name__ == "__main__":
    unittest.main()
