from cython.operator cimport dereference as deref
from vocab cimport vocab

cdef class ngram:
    cdef Ngram *thisptr

    def __cinit__(self, vocab v, unsigned order = defaultNgramOrder):
        self.thisptr = new Ngram(deref(<Vocab *>(v.thisptr)), order)

    def __dealloc__(self):
        del self.thisptr
