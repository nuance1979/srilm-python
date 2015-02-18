from c_vocab cimport VocabIndex
from cpython cimport array
from ngram cimport Stats
from common cimport Boolean

cdef class Lm:
    """Abstract class to encourage uniform interface"""
    def prob(self, VocabIndex word, context):
        raise NotImplementedError('Abstract class method')

    def prob_ngram(self, ngram):
        """Return log probability of p(ngram[-1] | ngram[-2], ngram[-3], ...)

           Noe that this function takes ngram in its *natural* order.
        """
        cdef VocabIndex word = ngram[-1]
        cdef array.array context = ngram[:-1].reverse()
        return self.prob(word, context)
    
    def read(self, const char *fname, Boolean limitVocab = False):
        raise NotImplementedError('Abstract class method')

    def write(self, const char *fname):
        raise NotImplementedError('Abstract class method')

    def train(self, Stats ts):
        raise NotImplementedError('Abstract class method')

    def test(self, Stats ns):
        raise NotImplementedError('Abstract class method')
