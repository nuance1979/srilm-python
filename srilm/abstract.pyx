from cython.operator cimport dereference as deref
from c_vocab cimport VocabIndex
from cpython cimport array
from ngram cimport Stats
from common cimport Boolean, File
from vocab cimport Vocab

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
    
    def read(self, const char *fname, Boolean limitVocab = False, binary = False):
        mode = 'rb' if binary else 'r'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        ok = self.lmptr.read(deref(fptr), limitVocab)
        del fptr
        return ok

    def write(self, const char *fname, binary = False):
        mode = 'wb' if binary else 'w'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        if binary:
            self.lmptr.writeBinary(deref(fptr))
        else:
            self.lmptr.write(deref(fptr))
        del fptr

    def train(self, Stats ts):
        raise NotImplementedError('Abstract class method')

    def test(self, Stats ns):
        raise NotImplementedError('Abstract class method')
