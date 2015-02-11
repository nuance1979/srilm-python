import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

from cython.operator cimport dereference as deref
from vocab cimport vocab
from cpython cimport array
from array import array
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

cdef inline bint _iswords(words):
    return isinstance(words, array) and words.typecode == 'I'

cdef inline void _tocstring(unsigned int order, VocabIndex *buff, array.array words):
    cdef int n = min(order, len(words))
    cdef int i
    for i in range(n):
        buff[i] = words[i]
    buff[n] = 0

cdef inline array.array _toarray(unsigned int order, VocabIndex *buff):
    cdef array.array a = array('I', [])
    cdef int i
    for i in range(order):
        a.append(buff[i])
    return a

cdef class lm:

    def __cinit__(self, vocab v, unsigned order = defaultNgramOrder):
        self.thisptr = new Ngram(deref(<Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError

    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.thisptr

    @property
    def order(self):
        return self.thisptr.setorder(0)

    @order.setter
    def order(self, unsigned neworder):
        self.thisptr.setorder(neworder)
        self.keysptr = <VocabIndex *>PyMem_Realloc(self.keysptr, (neworder+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError

    def wordProb(self, VocabIndex word, context):
        if context is None:
            return self.thisptr.wordProb(word, NULL)
        elif _iswords(context):
            _tocstring(self.order, self.keysptr, context)
            return self.thisptr.wordProb(word, self.keysptr)
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

cdef class stats:

    def __cinit__(self, vocab v, unsigned int order):
        self.thisptr = new NgramStats(deref(<Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError

    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.thisptr

    @property
    def order(self):
        return self.thisptr.getorder()

    def findCount(self, words):
        cdef NgramCount *p
        if words is None:
            p = self.thisptr.findCount(NULL)
            return 0 if p == NULL else deref(p)
        elif _iswords(words):
            _tocstring(self.order, self.keysptr, words)
            p = self.thisptr.findCount(self.keysptr)
            return 0 if p == NULL else deref(p)
        else:
            raise TypeError('Expect array')

    def insertCount(self, words, count = 1):
        cdef NgramCount *p
        if words is None:
            p = self.thisptr.insertCount(NULL)
            p[0] += count
        elif _iswords(words):
            _tocstring(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] += count
        else:
            raise TypeError('Expect array')

    def removeCount(self, words):
        cdef NgramCount count
        cdef Boolean b
        if words is None:
            b = self.thisptr.removeCount(NULL, &count)
            return count if b else 0
        elif _iswords(words):
            _tocstring(self.order, self.keysptr, words)
            b = self.thisptr.removeCount(self.keysptr, &count)
            return count if b else 0
        else:
            raise TypeError('Expect array')

    def sumCounts(self):
        return self.thisptr.sumCounts()

    def read(self, fname):
        cdef File *fptr
        cdef Boolean b
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'r', 1)
        b = self.thisptr.read(deref(fptr))
        del fptr
        return b

    def write(self, fname):
        cdef File *fptr
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'w', 1)
        self.thisptr.write(deref(fptr))
        del fptr

    def __getitem__(self, words):
        return self.findCount(words)

    def __setitem__(self, words, count):
        self.insertCount(words, count)

    def __delitem__(self, words):
        self.removeCount(words)

    def __iter__(self):
        self.iterptr = new NgramsIter(deref(self.thisptr), self.keysptr, self.order, NULL)
        return self

    def __next__(self):
        cdef NgramCount *p = self.iterptr.next()
        cdef array.array keys
        cdef NgramCount i
        if p == NULL:
            del self.iterptr
            raise StopIteration
        else:
            keys = _toarray(self.order, self.keysptr)
            i = p[0]
            return (keys, i)
