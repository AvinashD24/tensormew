import unittest

import numpy as np

from src.frontier_buffer import FrontierBuffer


class FrontierBufferTests(unittest.TestCase):
    def test_empty_sample_is_none(self):
        buf = FrontierBuffer()
        self.assertEqual(len(buf), 0)
        self.assertIsNone(buf.sample(np.random.default_rng(0)))

    def test_keeps_best_score_and_metadata_per_map(self):
        buf = FrontierBuffer()
        buf.add(b"low", (1, 2), 0, 1.0)
        buf.add(b"high", (1, 2), 3, 9.0)
        buf.add(b"later", (1, 2), 7, 4.0)
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf.sample(np.random.default_rng(0)), b"high")
        meta = buf.entries[0]["meta"]
        self.assertEqual(meta["map"], (1, 2))
        self.assertEqual(meta["badges"], 3)
        self.assertEqual(meta["score"], 9.0)

    def test_equal_score_keeps_the_first_snapshot(self):
        buf = FrontierBuffer()
        buf.add(b"first", (3, 4), 1, 5.0)
        buf.add(b"second", (3, 4), 2, 5.0)
        self.assertEqual(buf.sample(np.random.default_rng(0)), b"first")
        self.assertEqual(buf.entries[0]["meta"]["badges"], 1)

    def test_distinct_maps_are_both_stored(self):
        buf = FrontierBuffer()
        buf.add(b"a", (0, 1), 0, 1.0)
        buf.add(b"b", (2, 3), 1, 2.0)
        self.assertEqual(len(buf), 2)
        seen = {buf.sample(np.random.default_rng(i)) for i in range(40)}
        self.assertEqual(seen, {b"a", b"b"})

    def test_capacity_drops_the_lowest_score(self):
        buf = FrontierBuffer(capacity=2)
        buf.add(b"low", (0, 1), 0, 1.0)
        buf.add(b"mid", (0, 2), 0, 2.0)
        buf.add(b"high", (0, 3), 4, 3.0)
        self.assertEqual(len(buf), 2)
        rng = np.random.default_rng(0)
        seen = {buf.sample(rng) for _ in range(40)}
        self.assertEqual(seen, {b"mid", b"high"})

    def test_new_map_below_the_worst_is_ignored(self):
        buf = FrontierBuffer(capacity=2)
        buf.add(b"a", (8, 1), 0, 5.0)
        buf.add(b"b", (8, 2), 0, 6.0)
        buf.add(b"c", (8, 3), 0, 1.0)
        self.assertEqual(len(buf), 2)
        rng = np.random.default_rng(1)
        seen = {buf.sample(rng) for _ in range(40)}
        self.assertEqual(seen, {b"a", b"b"})

    def test_default_capacity_keeps_sixty_four_best(self):
        buf = FrontierBuffer()
        for i in range(70):
            buf.add(bytes([i]), (i, i + 1), i % 4, float(i))
        self.assertEqual(len(buf), 64)
        scores = sorted(entry["meta"]["score"] for entry in buf.entries)
        self.assertEqual(scores[0], 6.0)
        self.assertEqual(scores[-1], 69.0)


if __name__ == "__main__":
    unittest.main()
