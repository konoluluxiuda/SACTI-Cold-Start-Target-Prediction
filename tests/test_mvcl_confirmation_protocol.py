import copy
import unittest
from tools.seal_mvcl_confirmation import validate_policy, ROOT, read


class ConfirmationProtocolTests(unittest.TestCase):
    def setUp(self):
        self.policy = read(ROOT/'configs/mvcl_formal_confirmation.json')
        self.reference = read(ROOT/self.policy['matched_sacti_protocol'])

    def test_fixed_scope(self):
        validate_policy(self.policy, self.reference)

    def test_reject_selective_folds_or_unseen_claim(self):
        for key, value in [('folds', [1]), ('new_lockbox_claim', True),
                           ('reuse_development_checkpoint', True), ('dataset_specific_selection', True)]:
            policy = copy.deepcopy(self.policy)
            policy[key] = value
            with self.assertRaises(ValueError):
                validate_policy(policy, self.reference)


if __name__ == '__main__':
    unittest.main()
