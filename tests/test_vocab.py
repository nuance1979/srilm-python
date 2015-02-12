import unittest
import srilm

class TestVocab(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.vocab()

    def test_in(self):
        self.assertIn('<unk>', self.vocab)

    def test_add(self):
        self.vocab.add('xixi')
        self.assertIn('xixi', self.vocab)
    
    def test_delete(self):
        del self.vocab['<s>']
        self.assertNotIn('<s>', self.vocab)

    def test_get(self):
        self.vocab.add('xixi')
        a = self.vocab['xixi']
        self.assertEqual(self.vocab[a], 'xixi')

    def test_iter(self):
        for w, i in self.vocab:
            self.assertEqual(self.vocab[w], i)
            self.assertEqual(self.vocab[i], w)

    def tearDown(self):
        del self.vocab

if __name__ == '__main__':
    unittest.main()
