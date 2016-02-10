import unittest
import srilm.vocab
import srilm.stats
import srilm.discount
import array
import random
import tempfile
import os

class TestNgramDiscount(unittest.TestCase):

    def setUp(self):
        self.discount = srilm.discount.Discount(method='kneser-ney', interpolate=True)

    def test_init(self):
        self.assertEqual(self.discount.method, 'kneser-ney')
        self.assertTrue(self.discount.interpolate)
        self.assertRaises(ValueError, srilm.discount.Discount, 'xixi-haha')
        with self.assertRaises(ValueError) as cm:
            d = srilm.discount.Discount(method='kneser-ney', discount='haha')
        self.assertEqual(type(cm.exception), ValueError)

    def test_read_write(self):
        self.discount.discount = 0.1
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.discount.write(fname)
        d = srilm.discount.Discount()
        d.read(fname)
#        self.assertEqual(d.discount, 0.1)
#        self.assertEqual(d.method, 'kneser-ney')
        os.remove(fname)

    def test_estimate(self):
        text = """
It was the best of times,
it was the worst of times,
it was the age of wisdom,
it was the age of foolishness,
it was the epoch of belief,
it was the epoch of incredulity, it was the season of Light,
it was the season of Darkness, it was the spring of hope,
it was the winter of despair,
"""
        v = srilm.vocab.Vocab()
        ts = srilm.stats.Stats(v, 3)
        for w in text.split():
            v.add(w)
        ts.count_string(text)
        self.assertTrue(self.discount.estimate(ts, 3))
        self.assertAlmostEqual(self.discount.discount, 0.6862745098039216)

    def test_read_write(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.discount.discount = 0.02
        self.discount.write(fname)
        d = srilm.discount.Discount()
        d.read(fname)
        self.assertEqual(self.discount.method, d.method)
        self.assertEqual(self.discount.interpolate, d.interpolate)
        self.assertEqual(self.discount.discount, d.discount)
        os.remove(fname)

    def tearDown(self):
        del self.discount

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNgramDiscount)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite]))
