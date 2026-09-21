import json
from pathlib import Path
import unittest

import pandas as pd

from scripts.prepare_data import split_posts
from scripts.metrics import token_repetition_ratio
from utils.io import load_config
from utils.text import clean_text, continuation_ids, training_labels

ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    def posts(self):
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=20, tz="UTC"),
            "text": [f"Synthetic post number {i}. Keep this punctuation!" for i in range(20)],
        })

    def test_cleaning_keeps_style_but_removes_links(self):
        self.assertEqual(clean_text("Hi, @name! https://example.com Fine?"), "Hi, ! Fine?")

    def test_splits_are_chronological(self):
        parts = split_posts(self.posts().sample(frac=1, random_state=4), min_chars=20)
        self.assertEqual([len(x) for x in parts.values()], [16, 2, 2])
        self.assertLess(parts["train"].date.max(), parts["val"].date.min())
        self.assertLess(parts["val"].date.max(), parts["test"].date.min())
        self.assertEqual(set(parts["train"].columns), {"date", "cleaned_text"})

    def test_duplicates_do_not_cross_splits(self):
        posts = self.posts()
        duplicate = posts.iloc[[0]].copy()
        duplicate["date"] = pd.Timestamp("2026-01-01", tz="UTC")
        joined = pd.concat(split_posts(pd.concat([posts, duplicate]), min_chars=20).values())
        self.assertEqual(len(joined), 20)
        self.assertFalse(joined.cleaned_text.duplicated().any())

    def test_equal_timestamps_stay_together(self):
        posts = self.posts()
        posts.loc[16, "date"] = posts.loc[15, "date"]
        parts = split_posts(posts, min_chars=20)
        self.assertLess(parts["train"].date.max(), parts["val"].date.min())

    def test_invalid_dates_fail(self):
        posts = self.posts()
        posts.loc[0, "date"] = pd.NaT
        with self.assertRaises(ValueError):
            split_posts(posts, min_chars=20)

    def test_small_input_fails(self):
        with self.assertRaises(ValueError):
            split_posts(self.posts().iloc[:5], min_chars=20)

    def test_padding_is_not_a_training_target(self):
        self.assertEqual(training_labels([[10, 2, 2]], [[1, 1, 0]]), [[10, 2, -100]])

    def test_continuation_uses_token_boundary(self):
        self.assertEqual(continuation_ids([1, 5, 8, 9], 2), [8, 9])

    def test_repetition_definition(self):
        self.assertAlmostEqual(token_repetition_ratio("a a b"), 0.5)

    def test_config_names_do_not_collide(self):
        names = [str(x.relative_to(ROOT)).lower() for x in (ROOT / "configs").rglob("*.json")]
        self.assertEqual(len(names), len(set(names)))

    def test_base_config_is_flat(self):
        self.assertIn("train_data", load_config(ROOT / "configs/base.json"))

    def test_credentials_are_empty(self):
        cfg = json.loads((ROOT / "configs/telegram_config.json").read_text())
        for key in ("api_id", "api_hash", "phone", "code", "password"):
            self.assertFalse(cfg[key])


if __name__ == "__main__":
    unittest.main()
