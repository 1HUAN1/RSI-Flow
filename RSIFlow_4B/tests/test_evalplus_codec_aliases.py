import unittest

from sia.task_meta.evalplus_codec import UnsupportedWireValue, decode, encode


class EvalPlusCodecAliasesTest(unittest.TestCase):
    def test_mbpp_repeated_dictionary_retains_identity(self):
        original = [{}] * 5
        restored = decode(encode(original))
        self.assertEqual(restored, original)
        self.assertTrue(all(value is restored[0] for value in restored))
        restored[0]["changed"] = 1
        self.assertEqual(restored[-1]["changed"], 1)

    def test_nested_alias_and_distinct_container(self):
        shared = [1]
        original = ({"a": shared}, shared, [1])
        restored = decode(encode(original))
        self.assertIs(restored[0]["a"], restored[1])
        self.assertIsNot(restored[1], restored[2])
    def test_empty_tuple_before_alias_keeps_reference_numbering(self):
        shared = {"x": 1}
        original = ((), shared, shared)
        restored = decode(encode(original))
        self.assertEqual(restored[0], ())
        self.assertIs(restored[1], restored[2])


    def test_cycles_remain_rejected(self):
        cycle = []
        cycle.append(cycle)
        with self.assertRaises(UnsupportedWireValue):
            encode(cycle)
        with self.assertRaises(UnsupportedWireValue):
            decode({"t": "list", "v": [{"t": "ref", "v": 0}]})

    def test_bad_reference_remains_rejected(self):
        for reference in (-1, 0, True, "0"):
            with self.subTest(reference=reference):
                with self.assertRaises(UnsupportedWireValue):
                    decode({"t": "ref", "v": reference})
