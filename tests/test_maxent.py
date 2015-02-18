import unittest
import srilm
import array
import random
import tempfile
import os

class TestMaxentLm(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.ngram.Stats(self.vocab, 3)
        self.lm = srilm.maxent.Lm(self.vocab, 3)

    def test_order(self):
        self.assertEqual(self.lm.order, 3)

    def tearDown(self):
        del self.lm
        del self.stats
        del self.vocab

if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestMaxentLm)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite1]))
