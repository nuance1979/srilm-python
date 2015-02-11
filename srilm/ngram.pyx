from cython.operator cimport dereference as deref
from vocab cimport vocab
from cpython cimport array
from array import array

cdef class ngram:
    cdef Ngram *thisptr

    def __cinit__(self, vocab v, unsigned order = defaultNgramOrder):
        self.thisptr = new Ngram(deref(<Vocab *>(v.thisptr)), order)

    def __dealloc__(self):
        del self.thisptr

    @property
    def order(self):
        return self.thisptr.setorder(0)

    @order.setter
    def order(self, unsigned neworder):
        self.thisptr.setorder(neworder)

    def wordProb(self, VocabIndex word, context):
        if context is None:
            return self.thisptr.wordProb(word, NULL)
        elif isinstance(context, array) and context.typecode == 'i':
            return self.thisptr.wordProb(word, (<array.array>context).data.as_uints)
        else:
            raise TypeError('Expect array')

    def read(self, fname, Boolean limitVocab = 0):
        cdef File *fptr
        cdef Boolean ok
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'r', 1)
        ok = self.thisptr.read(deref(fptr), limitVocab)
        del fptr
        return ok

    def write(self, fname):
        cdef File *fptr
        cdef Boolean ok
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'w', 1)
        ok = self.thisptr.write(deref(fptr))
        del fptr
        return ok
