"""
Tests for the AA Trade Evaluator core and memory layers.

Run:  python test_wl.py

These tests do not need Discord or a config file. They use a temporary value
list and a temporary memory database, so your real data is never touched.
"""

import math
import os
import shutil
import tempfile
import time
import unittest

import wl_core as core
import wl_memory as mem

SAMPLE_VALUE_LIST = """# sample value list for tests
215 | Shiny Hunter / Hunter Shiny
5000 | Dio
500 | Value
1000 | Goku
3000 | Hie / Hiei (SHINY)
2000 | Hie / Hiei
"""

TMP = tempfile.mkdtemp(prefix="wl_test_")


def load_sample_catalog():
    path = os.path.join(TMP, "value_list.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(SAMPLE_VALUE_LIST)
    core.VALUE_LIST_FILE = path
    core.load_value_list()


def use_temp_memory():
    mem.MEMORY_FILE = os.path.join(TMP, f"memory_{time.time_ns()}.db")
    mem.init()


class ParserTests(unittest.TestCase):

    def test_single_item_lookup(self):
        self.assertEqual(core.split_sides("is shiny sakamoto"), (["shiny sakamoto"], []))

    def test_my_on_left(self):
        self.assertEqual(
            core.split_sides("my shiny cream and shiny somber for homura"),
            (["shiny cream", "shiny somber"], ["homura"]))

    def test_my_on_right_reads_correctly(self):
        self.assertEqual(
            core.split_sides("is Gohan for my Goku win or lose?"),
            (["goku"], ["gohan"]))

    def test_chat_post_my_on_right(self):
        self.assertEqual(core.split_sides("!wl Goku for my Gohan"), (["gohan"], ["goku"]))

    def test_plus_sign_with_my(self):
        self.assertEqual(core.split_sides("is my Goku + Dio for Gohan"),
                         (["goku", "dio"], ["gohan"]))

    def test_no_my_means_left_is_give(self):
        self.assertEqual(core.split_sides("is Goku for Gohan"), (["goku"], ["gohan"]))

    def test_mamy_is_not_my(self):
        self.assertEqual(core.split_sides("is mamy for gohan"), (["mamy"], ["gohan"]))

    def test_bare_command_returns_none(self):
        self.assertIsNone(core.split_sides("is"))

    def test_norm_key_quantity_and_shiny(self):
        self.assertEqual(core.norm_key("2x Shiny Sakamoto!"), "sakamoto shiny")

    def test_strip_filler_question_words(self):
        self.assertEqual(core.strip_filler(core.norm_key("what is dio value")), "dio")

    def test_strip_filler_tagalog_words(self):
        self.assertEqual(core.strip_filler(core.norm_key("ano value ni dio")), "dio")


class CatalogTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_sample_catalog()

    def test_plain_name_resolves(self):
        self.assertEqual(core.resolve_query("dio"), ("ok", "dio", []))

    def test_question_wording_resolves(self):
        self.assertEqual(core.resolve_query("what is dio value"), ("ok", "dio", []))

    def test_tagalog_wording_resolves(self):
        self.assertEqual(core.resolve_query("ano value ni dio"), ("ok", "dio", []))

    def test_unit_named_value_still_resolves(self):
        # "value" is also a filler word, but the full text is tried first
        self.assertEqual(core.resolve_query("value"), ("ok", "value", []))

    def test_shiny_lookup_resolves(self):
        status, name, _ = core.resolve_query("shiny hie")
        self.assertEqual((status, name), ("ok", "hiei shiny"))

    def test_shiny_estimate_for_unlisted_shiny(self):
        self.assertTrue(core.is_known("goku shiny"))
        self.assertIsNotNone(core.shiny_estimate("goku shiny"))

    def test_unknown_item(self):
        status, name, opts = core.resolve_query("xyzzy")
        self.assertEqual(status, "unknown")
        self.assertIsNone(name)

    def test_personal_choice_overrides_catalog(self):
        lookup = lambda k: "goku" if k == "hiei" else None
        self.assertEqual(core.resolve_query("hiei", lookup), ("ok", "goku", []))

    def test_choice_saved_under_stripped_key_applies_to_wording(self):
        lookup = lambda k: "goku" if k == "dio" else None
        self.assertEqual(core.resolve_query("what is dio value", lookup), ("ok", "goku", []))

    def test_learn_key_strips_filler(self):
        self.assertEqual(core.learn_key("what is dio value"), "dio")


class MemoryTests(unittest.TestCase):

    def setUp(self):
        use_temp_memory()

    def test_personal_choice_round_trip(self):
        mem.learn("u1", "dio", "goku")
        self.assertEqual(mem.learned_map("u1"), {"dio": "goku"})

    def test_consensus_needs_three_different_users(self):
        mem.learn("u1", "dio", "dio")
        mem.learn("u2", "dio", "dio")
        self.assertIsNone(mem.consensus("dio"))
        mem.learn("u3", "dio", "dio")
        self.assertEqual(mem.consensus("dio"), "dio")

    def test_same_user_repeating_does_not_build_consensus(self):
        for _ in range(10):
            mem.learn("u1", "dio", "dio")
        self.assertIsNone(mem.consensus("dio"))

    def test_consensus_requires_clear_majority(self):
        for uid in ("u1", "u2", "u3"):
            mem.learn(uid, "dio", "dio")
        for uid in ("u4", "u5", "u6"):
            mem.learn(uid, "dio", "goku")
        self.assertIsNone(mem.consensus("dio"))

    def test_last_trade_round_trip(self):
        mem.set_last_trade("u1", ["goku"], ["dio"])
        self.assertEqual(mem.get_last_trade("u1"), {"give": ["goku"], "get": ["dio"]})

    def test_forget_removes_user_data(self):
        mem.learn("u1", "dio", "dio")
        mem.log_message("u1", "hi")
        mem.set_last_trade("u1", ["goku"], ["dio"])
        mem.forget("u1")
        self.assertEqual(mem.learned_map("u1"), {})
        self.assertIsNone(mem.get_last_trade("u1"))

    def test_message_history_is_trimmed(self):
        original = mem.MAX_MESSAGES_PER_USER
        mem.MAX_MESSAGES_PER_USER = 5
        try:
            for i in range(12):
                mem.log_message("u1", f"msg {i}")
            with mem.memory_conn() as c:
                count = c.execute("SELECT COUNT(*) FROM messages WHERE uid = 'u1'").fetchone()[0]
            self.assertEqual(count, 5)
        finally:
            mem.MAX_MESSAGES_PER_USER = original


class ModelTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        load_sample_catalog()

    def tearDown(self):
        core.TRADE_DB.clear()
        core.recalculate()

    def test_recalculate_produces_positive_values(self):
        core.TRADE_DB["1"] = {"id": "1", "offered": ["goku"], "requested": ["dio"],
                              "ups": 3, "downs": 0, "timestamp": "", "author": "a",
                              "ts": time.time()}
        core.recalculate()
        self.assertIn("goku", core.VALUES)
        self.assertTrue(core.value_of("dio") > 0)
        self.assertTrue(math.isfinite(core.value_of("goku")))

    def test_backtest_needs_enough_votes(self):
        core.TRADE_DB.clear()
        self.assertIsNone(core.backtest())

    def test_direct_votes_count_exact_matches_once_per_author(self):
        now = time.time()
        for i in range(3):
            core.TRADE_DB[str(i)] = {"id": str(i), "offered": ["goku"], "requested": ["dio"],
                                     "ups": 2, "downs": 0, "timestamp": "",
                                     "author": "same", "ts": now}
        wins, losses = core.direct_votes(["goku"], ["dio"])
        self.assertEqual((wins, losses), (1, 0))


if __name__ == "__main__":
    try:
        unittest.main(verbosity=2)
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
