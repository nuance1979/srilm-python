import unittest
import srilm

class TestVocab(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.vocab()

    def test_add_delete(self):
        self.assertEqual(len(self.vocab), 4)
        for i in range(1000):
            self.vocab.addWord('word%d' % i)
        self.assertTrue('word15' in self.vocab)
        for i in range(500):
            w = 'word%i' % i
            if i % 2 == 0:
                del self.vocab[w]
            else:
                self.vocab.remove(w)
        self.assertIsNone(self.vocab['word45'])
        a = self.vocab['word500']
        self.assertEqual(self.vocab[a], 'word500')
        self.assertEqual(len(self.vocab), 504) 

    def tearDown(self):
        del self.vocab

if __name__ == '__main__':
    unittest.main()
