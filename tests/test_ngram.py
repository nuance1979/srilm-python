import unittest
import srilm.vocab
import srilm.ngram
import srilm.stats
import srilm.discount
import srilm.utils
import tempfile
import os


class TestNgramLM(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.lm = srilm.ngram.Lm(self.vocab, 3)
        self.stats = srilm.stats.Stats(self.vocab, 3)

    def test_order(self):
        self.assertEqual(self.lm.order, 3)
        with self.assertRaises(AttributeError) as cm:
            self.lm.order = 4
        self.assertEqual(type(cm.exception), AttributeError)

    def test_len(self):
        self.assertEqual(len(self.lm), 0)

    def test_prob(self):
        self.assertEqual(self.lm.prob_ngram(self.vocab.index(['it', 'was', 'the'])), float('-Inf'))

    def test_compare_with_command_line(self):
        # reference was created with this command line
        # cmd = '../bin/i686-m64/ngram-count -order 3 -vocab tests/98c1v.txt -unk -ukndiscount -interpolate -lm tests/lm.txt -text tests/98c1.txt -gt3min 1'
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')
        for i in range(3):
            self.lm.set_discount(i + 1, srilm.discount.Discount(method='kneser-ney', interpolate=True))
        self.lm.train(self.stats)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.lm.write(fname)
        with open(fname) as f:
            out_text_lm = f.read()
        with open('tests/lm.txt') as f:
            ref_text_lm = f.read()
        self.assertEqual(out_text_lm, ref_text_lm)
        os.remove(fname)

    def tearDown(self):
        del self.stats
        del self.lm
        del self.vocab


class TestNgramLMInDepth(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.lm = srilm.ngram.Lm(self.vocab, 3)
        self.stats = srilm.stats.Stats(self.vocab, 3)
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
        for w in text.split():
            self.vocab.add(w)
        self.stats.count_string(text)
        for i in range(1, 4):
            self.lm.set_discount(i, srilm.discount.Discount(method='kneser-ney'))
        self.assertTrue(self.lm.train(self.stats))

    def test_train(self):
        self.assertAlmostEqual(self.lm.prob_ngram(self.vocab.index(['it', 'was', 'the'])), -2.5774917602539062)

    def test_test(self):
        prob, denom, ppl = self.lm.test(self.stats)
        self.assertAlmostEqual(ppl, 10.253298042321083)
        s = srilm.stats.Stats(self.vocab, 2)
        prob, denom, ppl = self.lm.test(s)
        self.assertEqual(str(ppl), 'nan')

    def test_mix(self):
        lm = srilm.ngram.Lm(self.vocab, 3)
        lm.read('tests/lm.txt')
        lm.mix_lm(self.lm, 0.5)
        self.assertAlmostEqual(lm.prob_ngram(self.vocab.index('how are you'.split())), -0.44557836651802063)

    def test_prune(self):
        lm = srilm.ngram.Lm(self.vocab, 2)
        lm.read('tests/lm.txt')
        self.assertEqual(len(self.lm), 44)
        self.lm.prune(0.0001, 2, lm)
        self.assertEqual(len(self.lm), 16)

    def test_read_write(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.lm.write(fname)
        lm = srilm.ngram.Lm(self.vocab, 3)
        lm.read(fname)
        b = self.vocab.index('it was the'.split())
        self.assertAlmostEqual(self.lm.prob_ngram(b), lm.prob_ngram(b), 5)
        os.remove(fname)

    def test_iter(self):
        for c, i in self.lm:
            self.assertEqual(len(c), 2)
            for w, p in i:
                self.assertEqual(self.lm.prob(w, c), p)
        for c, i in self.lm.iter(1):
            self.assertEqual(len(c), 1)

    def test_rand_gen(self):
        srilm.utils.rand_seed(1000)
        ans = ['was', 'the', 'winter', '<unk>', 'it', 'was', 'the', '<unk>', '<unk>']
        self.assertEqual(self.lm.rand_gen(10), ans)

    def tearDown(self):
        del self.stats
        del self.lm
        del self.vocab


class TestNgramCountLM(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.lm = srilm.ngram.CountLm(self.vocab, 3)
        self.stats = srilm.stats.Stats(self.vocab, 3)
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')
        self.heldout = srilm.stats.Stats(self.vocab, 3)
        self.heldout.count_file('tests/98c2.txt')

    def test_train(self):
        self.assertTrue(self.lm.train(self.stats, self.heldout))

    def test_read_write(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.lm.write(fname)
        self.lm.read(fname)
        b = self.vocab.index('it was the'.split())
        self.assertEqual(self.lm.prob_ngram(b), -2.033423662185669)
        os.remove(fname)

    def test_prob(self):
        self.assertTrue(self.lm.train(self.stats, self.heldout))
        b = self.vocab.index('it was the'.split())
        self.assertAlmostEqual(self.lm.prob_ngram(b), -1.2050670385360718)

    def tearDown(self):
        del self.heldout
        del self.stats
        del self.lm
        del self.vocab


class TestNgramSimpleClassLM(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.stats.Stats(self.vocab, 2)
        self.lm = srilm.ngram.SimpleClassLm(self.vocab, 2)
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')

    def test_order(self):
        self.assertEqual(self.lm.order, 2)

    def test_train_class(self):
        ts = srilm.stats.Stats(self.vocab, 2)
        ts.count_file('tests/98c1.txt')
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        fd, fcname = tempfile.mkstemp()
        os.close(fd)
        srilm.utils.train_class(ts, 5, fname, fcname, 'inc')
        os.remove(fname)
        os.remove(fcname)

    def test_train(self):
        ts = srilm.stats.Stats(self.vocab, 2)
        ts.count_file('tests/98c1.txt')
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        fd, fcname = tempfile.mkstemp()
        os.close(fd)
        srilm.utils.train_class(ts, 5, fname, fcname, 'inc')
        self.lm.train(fname, fcname)
        os.remove(fname)
        os.remove(fcname)

    def tearDown(self):
        del self.lm
        del self.stats
        del self.vocab


class TestNgramCacheLM(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.stats.Stats(self.vocab, 3)
        self.lm = srilm.ngram.CacheLm(self.vocab, 10)
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')

    def test_length(self):
        self.assertEqual(self.lm.length, 10)

    def test_prob(self):
        for w, i in self.stats.iter(2):
            self.lm.prob_ngram(w)
        self.assertAlmostEqual(self.lm.prob(self.vocab['<unk>'], None), -0.6989700198173523)

    def tearDown(self):
        del self.lm
        del self.stats
        del self.vocab

if __name__ == '__main__':
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestNgramLM)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestNgramLMInDepth)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestNgramCountLM)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestNgramSimpleClassLM)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(TestNgramCacheLM)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite2, suite3, suite4, suite5, suite6]))
