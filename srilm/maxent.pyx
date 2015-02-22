from cython.operator cimport dereference as deref
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from vocab cimport Vocab
from ngram cimport defaultNgramOrder, Stats
cimport ngram

cdef class Lm(base.Lm):
    """Maximum Entropy Language Model"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        if order < 1:
            raise ValueError('Invalid order')
        self.thisptr = new MEModel(deref(v.thisptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr # to use shared methods

    def __dealloc__(self):
        del self.thisptr

    property order:
        def __get__(self):
            return self._order

    def train(self, Stats ts, alpha = 0.5, sigma2 = 6.0):
        return self.thisptr.estimate(deref(ts.thisptr), alpha, sigma2)

    def to_ngram_lm(self):
        cdef Ngram *p = self.thisptr.getNgramLM()
        cdef ngram.Lm new_lm = ngram.Lm(self._vocab, self._order)
        del new_lm.thisptr
        new_lm.thisptr = p
        new_lm.lmptr = <base.LM *>p # need to update lmptr as well!!!
        return new_lm
