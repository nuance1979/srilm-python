import unittest
import srilm
import array
import random

class TestNgramStats(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.vocab()
        self.stats = srilm.ngram.stats(self.vocab, 3)

    def test_get(self):
        words = array.array('I', [1,2,3])
        self.assertEqual(self.stats[words], 0)

    def test_set(self):
        words = array.array('I', [1,2,3])
        self.stats[words] = 100
        self.assertEqual(self.stats[words], 100)
    
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
        for w, i in self.stats:
            self.assertEqual(self.stats[w], i)

    def tearDown(self):
        del self.stats
        del self.vocab

class TestNgramLM(unittest.TestCase):
    
    def setUp(self):
        self.vocab = srilm.vocab.vocab()
        self.lm = srilm.ngram.lm(self.vocab, 3)

    def test_order(self):
        self.lm.order = 4
        self.assertEqual(self.lm.order, 4)

    def test_prob(self):
        pass

    def test_train(self):
        pass

    def test_eval(self):
        pass

    def test_read_write(self):
        pass

    def tearDown(self):
        del self.lm
        del self.vocab


if __name__ == '__main__':
    suite1 = unittest.TestLoader().loadTestsFromTestCase(TestNgramStats)
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestNgramLM)
    unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([suite1, suite2]))
