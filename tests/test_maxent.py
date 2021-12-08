import unittest
import srilm.vocab
import srilm.stats
import srilm.maxent
import tempfile
import os


class TestMaxentLm(unittest.TestCase):
    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.stats.Stats(self.vocab, 3)
        self.lm = srilm.maxent.Lm(self.vocab, 3)
        self.vocab.read("tests/98c1v.txt")
        self.stats.count_file("tests/98c1.txt")

    def test_order(self):
        self.assertEqual(self.lm.order, 3)

    def test_prob(self):
        self.assertTrue(self.lm.train(self.stats))
        self.assertAlmostEqual(
            self.lm.prob_ngram(self.vocab.index("it was the".split())),
            -1.2563170194625854,
        )

    def test_read_write(self):
        self.assertTrue(self.lm.train(self.stats))
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.lm.write(fname)
        lm = srilm.maxent.Lm(self.vocab, 3)
        lm.read(fname)
        b = self.vocab.index("it was the".split())
        self.assertEqual(self.lm.prob_ngram(b), lm.prob_ngram(b))
        os.remove(fname)

    def test_train_test(self):
        self.assertTrue(self.lm.train(self.stats))
        self.stats.make_test()
        _, _, ppl = self.lm.test(self.stats)
        self.assertAlmostEqual(ppl, 6.400844395613356)

    def test_to_ngram_lm(self):
        self.assertTrue(self.lm.train(self.stats))
        lm = self.lm.to_ngram_lm()
        b = self.vocab.index("it was the".split())
        self.assertEqual(self.lm.prob_ngram(b), lm.prob_ngram(b))

    def tearDown(self):
        del self.lm
        del self.stats
        del self.vocab


if __name__ == "__main__":
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestMaxentLm)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite1]))
