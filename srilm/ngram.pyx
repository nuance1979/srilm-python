from cython.operator cimport dereference as deref
cimport c_vocab
from c_vocab cimport Vocab_None
from vocab cimport Vocab
from stats cimport Stats
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from discount cimport ModKneserNey, KneserNey, GoodTuring, WittenBell, Discount
from array import array
from common cimport _fill_buffer_with_array, _create_array_from_buffer
import os
import tempfile

cdef class Lm(base.Lm):
    """Ngram language model"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        if order < 1:
            raise ValueError('Invalid order')
        self.thisptr = new Ngram(deref(v.thisptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr # to use shared methods
        self.dlistptr = <c_discount.Discount **>PyMem_Malloc(order * sizeof(c_discount.Discount *))
        if self.dlistptr == NULL:
            raise MemoryError
        cdef unsigned int i
        for i in range(order):
            self.dlistptr[i] = NULL
        self._dlist = []

    def __dealloc__(self):
        PyMem_Free(self.dlistptr)
        del self.thisptr

    def __len__(self):
        return self.thisptr.numNgrams(self.order)

    def set_discount(self, unsigned int order, Discount d):
        if order > self.order or order < 1:
            raise ValueError('Invalid order')
        if d.thisptr == NULL:
            raise ValueError('Corrupted Discount object')
        self.dlistptr[order - 1] = d.thisptr
        self._dlist.append(d) # keep a python reference to d

    def train(self, Stats ts):
        cdef bint b
        cdef int i
        for i in range(self.order):
            if self.dlistptr[i] == NULL:
                raise RuntimeError('Discount for order %d is not set yet; use set_discount()' % i+1)
        for i in range(self.order):
            if self._dlist[i].discount is None: 
                self._dlist[i].estimate(ts, i+1)
        b = self.thisptr.estimate(deref(ts.thisptr), self.dlistptr)
        return b

    def mix_lm(self, Lm in_lm, double in_weight):
        """Mix a ngram.Lm into this model with weight"""
        self.thisptr.mixProbs(deref(in_lm.thisptr), in_weight)

    def iter(self, unsigned int length):
        if length > self.order - 1:
            raise ValueError('Invalid context length')
        return _create_iter_context(self.thisptr, length)

    def __iter__(self):
        return self.iter(self.order - 1)

cdef LmIterContext _create_iter_context(Ngram *lmptr, unsigned int order):
    it = LmIterContext()
    it.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex)) # iterator should manage its own buffer
    if it.keysptr == NULL:
        raise MemoryError
    it.iterptr = new NgramBOsIter(deref(lmptr), it.keysptr, order, NULL)
    if it.iterptr == NULL:
        raise MemoryError
    it._iter_order = order
    return it

cdef class LmIterContext:
    """LM context iterator"""
    def __dealloc__(self):
        del self.iterptr
        PyMem_Free(self.keysptr)

    def __iter__(self):
        return self

    def __next__(self):
        cdef BOnode *p = self.iterptr.next()
        if p == NULL:
            raise StopIteration
        else:
            keys = _create_array_from_buffer(self._iter_order, self.keysptr)
            h = _create_iter_prob(p)
            return (keys, h)

cdef LmIterProb _create_iter_prob(BOnode *p):
    it = LmIterProb()
    it.iterptr = new NgramProbsIter(deref(p), NULL)
    if it.iterptr == NULL:
        raise MemoryError
    return it

cdef class LmIterProb:
    """LM probability iterator"""
    def __dealloc__(self):
        del self.iterptr

    def __iter__(self):
        return self
    
    def __next__(self):
        cdef VocabIndex word
        cdef LogP *p = self.iterptr.next(word)
        if p == NULL:
            raise StopIteration
        else:
            return (word, deref(p))

cdef class CountLm(base.Lm):
    """Ngram language model with deleted interpolation, a.k.a. Jelinek-Mercer, smoothing"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        if order < 1:
            raise ValueError('Invalid order')
        self.thisptr = new NgramCountLM(deref(v.thisptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr # to use shared methods

    def __dealloc__(self):
        del self.thisptr

    def train(self, Stats ts, max_iter = 100, min_delta = 0.001):
        self.thisptr.maxEMiters = max_iter
        self.thisptr.minEMdelta = min_delta
        return self.thisptr.estimate(deref(ts.thisptr))

cdef class SimpleClassLm(base.Lm):
    """Simple bigram class-based language model, where a word belongs to a unique class

    That is, p(w_0 | w_-1) = p(w_0 | c_0) p(c_0 | c_-1)
    """
    def __cinit__(self, Vocab v, unsigned order = 2):
        if order != 2:
            raise ValueError('Invalid order; expect 2')
        self._class_vocab_ptr = new SubVocab(deref(v.thisptr), False)
        if self._class_vocab_ptr == NULL:
            raise MemoryError
        self.thisptr = new SimpleClassNgram(deref(v.thisptr), deref(self._class_vocab_ptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr

    def __dealloc__(self):
        del self._class_vocab_ptr
        del self.thisptr

    def read_class(self, const char *fname):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr == NULL:
            raise MemoryError
        ok = self.thisptr.readClasses(deref(fptr))
        del fptr
        return ok

    def write_class(self, const char *fname):
        cdef File *fptr = new File(fname, 'w', 0)
        if fptr == NULL:
            raise MemoryError
        self.thisptr.writeClasses(deref(fptr))
        del fptr

    def train(self, const char *classes, const char *class_counts):
        """Train with bigram class counts and class definition"""
        self.read_class(classes)
        cdef Stats ts = Stats(self._vocab, 2, open_vocab = True)
        ts.read(class_counts)
        cdef c_discount.Discount **dlistptr = <c_discount.Discount **>PyMem_Malloc(2 * sizeof(c_discount.Discount *))
        cdef int i
        cdef Discount d
        for i in range(2):
            d = Discount(method='good-turing')
            d.estimate(ts, i+1)
            print d.discount
            dlistptr[i] = d.thisptr
            d.thisptr = NULL # transfer ownership
        b = (<Ngram *>self.thisptr).estimate(deref(ts.thisptr), dlistptr)
        for i in range(2):
            del dlistptr[i]
        PyMem_Free(dlistptr)
        return b

cdef class CacheLm(base.Lm):
    """Unigram cache language model"""
    def __cinit__(self, Vocab v, unsigned historyLength):
        if historyLength < 1:
            raise ValueError('Invalid history length')
        self.thisptr = new CacheLM(deref(v.thisptr), historyLength)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr 
        self._length = historyLength
        self._order = 1
        self.running = True # very important and easy to miss
        
    def __dealloc__(self):
        del self.thisptr

    property length:
        def __get__(self):
            return self._length
