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

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNgramStats)
    unittest.TextTestRunner(verbosity=2).run(suite)
