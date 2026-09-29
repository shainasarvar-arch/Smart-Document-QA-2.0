import unittest

import faiss
import numpy as np

from app import DocumentChunk, EMBEDDING_DIMENSIONS, search_index


class SearchIndexTests(unittest.TestCase):
    def test_returns_nearest_chunk_with_page_metadata(self) -> None:
        vectors = np.zeros((2, EMBEDDING_DIMENSIONS), dtype=np.float32)
        vectors[0, 0] = 1.0
        vectors[1, 1] = 1.0
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSIONS)
        index.add(vectors)
        chunks = [
            DocumentChunk("matches", "report.pdf", 3),
            DocumentChunk("other", "notes.pdf", 9),
        ]

        matches = search_index(index, chunks, vectors[:1])

        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0][0].text, "matches")
        self.assertEqual(matches[0][0].filename, "report.pdf")
        self.assertEqual(matches[0][0].page_number, 3)
        self.assertAlmostEqual(matches[0][1], 1.0)


if __name__ == "__main__":
    unittest.main()
