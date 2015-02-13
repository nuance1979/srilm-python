import unittest
import srilm
import array

class TestVocab(unittest.TestCase):

    def setUp(self):
        self.vocab = srilm.vocab.Vocab()

    def test_in(self):
        self.assertIn('<unk>', self.vocab)

    def test_add(self):
        self.vocab.add('xixi')
        self.assertIn('xixi', self.vocab)
        self.assertRaises(TypeError, self.vocab.add, 123)
    
    def test_delete(self):
        del self.vocab['<s>']
        self.assertNotIn('<s>', self.vocab)

    def test_get(self):
        self.vocab.add('xixi')
        a = self.vocab['xixi']
        self.assertEqual(self.vocab[a], 'xixi')
        self.assertRaises(TypeError, self.vocab.__getitem__, [1,2,3])

    def test_iter(self):
        for i in range(1000):
            self.vocab.add('word%d' % i)
        for w, i in self.vocab:
            self.assertEqual(self.vocab[w], i)
            self.assertEqual(self.vocab[i], w)

    def test_index(self):
        a = ['<unk>', '<unk>', '<unk>']
        b = array.array('I', [0,0,0])
        self.assertEqual(self.vocab.index(a), b)
        a = ['xixi', 'haha']
        b = array.array('I', [self.vocab.unk, self.vocab.unk])
        self.assertEqual(self.vocab.index(a), b)

    def test_string(self):
        a = ['<unk>', '<unk>', '<unk>']
        b = array.array('I', [0,0,0])
        self.assertEqual(self.vocab.string(b), a)
        b = [1,3,100]
        self.assertRaises(IndexError, self.vocab.string, b)

    def tearDown(self):
        del self.vocab

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestVocab)
    unittest.TextTestRunner(verbosity=2).run(suite)
