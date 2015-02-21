import unittest
import srilm
import array
import random
import tempfile
import os

class TestNgramStats(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.ngram.Stats(self.vocab, 3)

    def test_order(self):
        self.assertEqual(self.stats.order, 3)
        with self.assertRaises(AttributeError) as cm:
            self.stats.order = 4
        self.assertEqual(type(cm.exception), AttributeError)

    def test_get(self):
        words = array.array('I', [1,2,3])
        self.assertEqual(self.stats[words], 0)
        self.assertRaises(TypeError, self.stats.__getitem__, [1,2,3])

    def test_set(self):
        words = array.array('I', [1,2,3])
        self.stats[words] = 100
        self.assertEqual(self.stats[words], 100)
        self.assertRaises(TypeError, self.stats.__setitem__, [1,2,3], 100)
    
    def test_add(self):
        words = array.array('I', [1,2,3])
        self.assertEqual(self.stats[words], 0)
        self.stats.add(words, 10)
        self.assertEqual(self.stats[words], 10)

    def test_remove(self):
        words = array.array('I', [1,2,3])
        self.stats[words] = 100
        del self.stats[words]
        self.assertEqual(self.stats[words], 0)

    def test_iter(self):
        for i in range(100):
            self.vocab.add('word%d' % i)
        words = array.array('I', [1,2,3])
        for i in range(1000):
            words[random.randint(0,2)] = random.randint(0,100) 
            self.stats[words] = random.randint(1,1000)
        for w, i in self.stats.iter(2):
            self.assertEqual(len(w), 2)
            self.assertEqual(self.stats[w], i)

    def test_read_write(self):
        words = array.array('I', [1,2,3])
        self.stats[words] = 15
        words = array.array('I', [1,2,0])
        self.stats[words] = 2
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.stats.write(fname)
        new_stats = srilm.ngram.Stats(self.vocab, 3)
        new_stats.read(fname)
        for w, i in self.stats:
            self.assertEqual(new_stats[w], i)
        for w, i in new_stats:
            self.assertEqual(self.stats[w], i)
        os.remove(fname)
        self.assertRaises(IOError, self.stats.read, '/path/to/foo')
        self.assertRaises(IOError, self.stats.write, '/i/do/not/exist')

    def test_read_write_binary(self):
        words = array.array('I', [1,2,3])
        self.stats[words] = 15
        words = array.array('I', [1,2,0])
        self.stats[words] = 2
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.stats.write(fname, binary=True)
        new_stats = srilm.ngram.Stats(self.vocab, 3)
        new_stats.read(fname, binary=True)
        for w, i in self.stats:
            self.assertEqual(new_stats[w], i)
        for w, i in new_stats:
            self.assertEqual(self.stats[w], i)
        os.remove(fname)
        self.assertRaises(IOError, self.stats.read, '/path/to/foo')
        self.assertRaises(IOError, self.stats.write, '/i/do/not/exist')

    def test_count_file(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        text = 'this is a test\n'
        with open(fname, 'w') as fout:
            fout.write('this is a test\n')
        for w in text.split():
            self.vocab.add(w)
        a = ['this', 'is', 'a']
        b = self.vocab.index(a)
        self.assertEqual(self.stats[b], 0)
        self.assertEqual(self.stats.count_file(fname), 6)
        self.assertEqual(self.stats[b], 1)
        os.remove(fname)

    def test_count_string(self):
        text = 'this is a test\n'
        for w in text.split():
            self.vocab.add(w)
        a = ['a', 'test', '</s>']
        b = self.vocab.index(a)
        self.assertEqual(self.stats[b], 0)
        self.assertEqual(self.stats.count_string(text), 6)
        self.assertEqual(self.stats[b], 1)

    def test_count(self):
        text = 'this is a test\n'
        words = text.split()
        for w in words:
            self.vocab.add(w)
        b = self.vocab.index(words)
        self.assertEqual(self.stats[b], 0)
        self.assertEqual(self.stats.count(b), 4)
        self.assertEqual(self.stats[b], 1)

    def test_len(self):
        self.assertEqual(len(self.stats), 0)
        a = array.array('I', [1,2,3])
        self.stats[a] = 3
        self.assertEqual(len(self.stats), 3)
        self.stats.count_string('this is a test')
        self.assertEqual(len(self.stats), 7)

    def test_sum(self):
        text = 'this is a test'
        for w in text.split():
            self.vocab.add(w)
        self.stats.count_string('this is a test')
        self.stats.sum()
        self.assertEqual(self.stats[self.vocab.index(['is','a'])], 1)

    def test_make_test(self):
        text = 'this is a test'
        for w in text.split():
            self.vocab.add(w)
        self.stats.count_string(text)
        b = self.vocab.index('is a'.split())
        self.assertEqual(self.stats[b], 1)
        s = self.stats.make_test()
        self.assertEqual(s[b], 0)

    def tearDown(self):
        del self.stats
        del self.vocab

class TestNgramLM(unittest.TestCase):
    
    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.lm = srilm.ngram.Lm(self.vocab, 3)
        self.stats = srilm.ngram.Stats(self.vocab, 3)

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
        cmd = '../bin/i686-m64/ngram-count -order 3 -vocab tests/98c1v.txt -unk -ukndiscount -interpolate -lm tests/lm.txt -text tests/98c1.txt -gt3min 1'
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')
        for i in range(3):
            self.lm.set_discount(i+1, srilm.discount.Discount(method='kneser-ney', interpolate=True))
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
        self.stats = srilm.ngram.Stats(self.vocab, 3)
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
        for i in range(1,4):
            self.lm.set_discount(i, srilm.discount.Discount(method='kneser-ney'))
        self.assertTrue(self.lm.train(self.stats))

    def test_train(self):
        self.assertAlmostEqual(self.lm.prob_ngram(self.vocab.index(['it','was','the'])), -2.5774917602539062)

    def test_test(self):
        prob, denom, ppl = self.lm.test(self.stats)
        self.assertAlmostEqual(ppl, 10.253298042321083)
        s = srilm.ngram.Stats(self.vocab, 2)
        prob, denom, ppl = self.lm.test(s)
        self.assertEqual(str(ppl), 'nan')

    def test_mix(self):
        lm = srilm.ngram.Lm(self.vocab, 3)
        lm.read('tests/lm.txt')
        lm.mix_lm(self.lm, 0.5)
        self.assertAlmostEqual(lm.prob_ngram(self.vocab.index('how are you'.split())), -0.44557836651802063)

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
                
    def tearDown(self):
        del self.stats
        del self.lm
        del self.vocab

class TestNgramCountLM(unittest.TestCase):
    
    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.lm = srilm.ngram.CountLm(self.vocab, 3)
        self.stats = srilm.ngram.Stats(self.vocab, 3)
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')
        self.stats.sum()

    def test_train(self):
        self.assertTrue(self.lm.train(self.stats))

    def test_read_write(self):
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        self.lm.write(fname)
        self.lm.read(fname)
        self.lm.train(self.stats)
        b = self.vocab.index('it was the'.split())
        self.assertAlmostEqual(self.lm.prob_ngram(b), -2.033423662185669)
        os.remove(fname)

    def tearDown(self):
        del self.stats
        del self.lm
        del self.vocab

class TestNgramClassLm(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.ngram.Stats(self.vocab, 3)
        self.lm = srilm.ngram.ClassLm(self.vocab, 3)
        self.vocab.read('tests/98c1v.txt')
        self.stats.count_file('tests/98c1.txt')

    def test_order(self):
        self.assertEqual(self.lm.order, 3)

    def tearDown(self):
        del self.lm
        del self.stats
        del self.vocab

class TestNgramCacheLm(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()
        self.stats = srilm.ngram.Stats(self.vocab, 3)
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
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestNgramStats)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestNgramLM)
    suite3 = unittest.TestLoader().loadTestsFromTestCase(TestNgramLMInDepth)
    suite4 = unittest.TestLoader().loadTestsFromTestCase(TestNgramCountLM)
    suite5 = unittest.TestLoader().loadTestsFromTestCase(TestNgramClassLM)
    suite6 = unittest.TestLoader().loadTestsFromTestCase(TestNgramCacheLM)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite1, suite2, suite3, suite4, suite5, suite6]))
