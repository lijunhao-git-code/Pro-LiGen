"""Synthetic edge-case tests; these records are never added to the experiments."""
import unittest
from offline_analysis import choose, correct_holm, paired


def candidate(name='A', qed=.8, sa=2., turn=4, position=1, eligible=True):
    return dict(canonical=name, QED=qed, SA_raw=sa, turn=turn,
                candidate_position=position, eligible=eligible)


class OfflineTests(unittest.TestCase):
    def test_empty_and_ineligible_are_no_delivery(self):
        self.assertIsNone(choose([]))
        self.assertIsNone(choose([candidate(eligible=False)]))

    def test_qed_precedes_sa_without_rounding(self):
        lower = candidate('A', .2666613820163968, 1.)
        higher = candidate('B', .26666138201639683, 4., position=2)
        self.assertEqual(choose([lower, higher]), higher)

    def test_exact_tie_uses_sa_then_canonical(self):
        self.assertEqual(choose([candidate('C', sa=2., position=2), candidate('B', sa=2.),
                                 candidate('A', sa=3., position=3)])['canonical'], 'B')

    def test_duplicate_last_occurrence(self):
        self.assertEqual(choose([candidate(), candidate(turn=5)])['turn'], 5)

    def test_turn_policy_and_first_eligible(self):
        first = candidate(qed=.4)
        last = candidate('B', qed=.9, turn=5)
        self.assertEqual(choose([last, first], 'turn4_QED'), first)
        self.assertEqual(choose([last, first], 'union_first_eligible'), first)

    def test_budget_and_statistical_unit_checks(self):
        with self.assertRaises(AssertionError):
            choose([candidate(position=4)])
        with self.assertRaises(AssertionError):
            paired([0.] * 125)
        self.assertEqual(paired([0.] * 25)['p_raw'], 1.)

    def test_holm_families_do_not_merge(self):
        rows = [dict(family='primary', p_raw=.3), dict(family='primary', p_raw=1.)]
        rows += [dict(family='auxiliary', p_raw=p) for p in [.0005, .003, .5, .5]]
        correct_holm(rows)
        for actual, expected in zip([r['p_holm'] for r in rows], [.6, 1., .002, .009, 1., 1.]):
            self.assertAlmostEqual(actual, expected, places=14)


if __name__ == '__main__':
    unittest.main()
