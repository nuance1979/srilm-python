import unittest
import srilm
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
        pass

    def tearDown(self):
        del self.discount

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNgramDiscount)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite]))
