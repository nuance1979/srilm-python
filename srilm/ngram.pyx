"""
Module contains garden variety of ngram language models
"""

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
        """Get the number of highest order ngram probabilites in the language model"""
        return self.thisptr.numNgrams(self.order)

    def set_discount(self, unsigned int order, Discount d):
        """Set the discount object for a specific ngram order

        Note that in theory, you can even 'mix-and-match' different type of discounts
        in the same language model.
        """
        if order > self.order or order < 1:
            raise ValueError('Invalid order')
        if d.thisptr == NULL:
            raise ValueError('Corrupted Discount object')
        self.dlistptr[order - 1] = d.thisptr
        self._dlist.append(d) # keep a python reference to d

    def train(self, Stats ts):
        """Traint the Ngram language model from ngram counts"""
        cdef int i
        for i in range(self.order):
            if self.dlistptr[i] == NULL:
                raise RuntimeError('Discount for order %d is not set yet; use set_discount()' % i+1)
        for i in range(self.order):
            if self._dlist[i].discount is None: 
                self._dlist[i].estimate(ts, i+1)
        return self.thisptr.estimate(deref(ts.thisptr), self.dlistptr)

    def prune(self, double threshold, unsigned min_order = 2, base.Lm history_lm = None):
        """Prune the Ngram language model with Entropy-based pruning, aka, Stolcke pruning"""
        if history_lm is None:
            self.thisptr.pruneProbs(threshold, min_order, NULL)
        else:
            self.thisptr.pruneProbs(threshold, min_order, history_lm.lmptr)

    def mix_lm(self, Lm in_lm, double in_weight):
        """Mix a ngram.Lm into this model with weight"""
        self.thisptr.mixProbs(deref(in_lm.thisptr), in_weight)

    def iter(self, unsigned int length):
        """Iterate through context/history of certain length

        Returns a tuple of (array_of_the_context, iterator_for_the_probs_under_this_context)
        """
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
        cdef VocabIndex word = 0
        cdef LogP *p = self.iterptr.next(word)
        if p == NULL:
            raise StopIteration
        else:
            return (word, deref(p))

cdef class CountLm(base.Lm):
    """Ngram language model with deleted interpolation, a.k.a. Jelinek-Mercer smoothing"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        if order < 1:
            raise ValueError('Invalid order')
        self.thisptr = new NgramCountLM(deref(v.thisptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr # to use shared methods

    def __dealloc__(self):
        del self.thisptr

    def train(self, Stats train, Stats heldout, max_iter = 100, min_delta = 0.001):
        """Train the Jelinek-Mercer-smoothed ngram language model from ngram counts

        Note that you *need* to use a different, usually much smaller, heldout ngram counts
        from the main train ngram counts.
        """
        self.thisptr.maxEMiters = max_iter
        self.thisptr.minEMdelta = min_delta
        # initialize the model with a temp file
        cdef NgramCount s = 0
        cdef NgramCount c
        for _, c in train.iter(1):
            s += c
        fd, fcname = tempfile.mkstemp()
        os.close(fd)
        train.write(fcname)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        with open(fname, 'w') as f:
            f.write("""
order {0}
mixweights 3
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
countmodulus 1
vocabsize {1}
totalcount {2}
counts {3}
            """.format(self.order, len(self._vocab), s, fcname))
        self.read(fname)
        ok = self.thisptr.estimate(deref(heldout.thisptr))
        os.remove(fname)
        os.remove(fcname)
        return ok

cdef class SimpleClassLm(base.Lm):
    """Simple bigram class-based language model, where a word belongs to a unique class"""
    def __cinit__(self, Vocab v, unsigned order = 2):
        if order != 2:
            raise ValueError('Invalid order; expect 2')
        self._class_vocab_ptr = new SubVocab(deref(v.thisptr), False)
        if self._class_vocab_ptr == NULL:
            raise MemoryError
        self.thisptr = new SimpleClassNgram(deref(v.thisptr), deref(self._class_vocab_ptr), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.lmptr = <base.LM *>self.thisptr # to use shared methods

    def __dealloc__(self):
        del self._class_vocab_ptr
        del self.thisptr

    def read_class(self, const char *fname):
        """Read class definition from a file

        In fact, the file defines a unigram language model of p(w | c).
        """
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr == NULL:
            raise MemoryError
        ok = self.thisptr.readClasses(deref(fptr))
        del fptr
        return ok

    def write_class(self, const char *fname):
        """Write class definition to a file

        In fact, the file defines a unigram language model of p(w | c).
        """
        cdef File *fptr = new File(fname, 'w', 0)
        if fptr == NULL:
            raise MemoryError
        self.thisptr.writeClasses(deref(fptr))
        del fptr

    def train(self, const char *classes, const char *class_counts):
        """Train the simple class-based language model from bigram class counts and class definition"""
        self.read_class(classes)
        cdef Stats ts = Stats(self._vocab, 2)
        ts.read(class_counts)
        cdef c_discount.Discount **dlistptr = <c_discount.Discount **>PyMem_Malloc(2 * sizeof(c_discount.Discount *))
        cdef int i
        cdef Discount d
        for i in range(2):
            d = Discount(method='kneser-ney')
            d.estimate(ts, i+1)
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
        """Length of the cache"""
        def __get__(self):
            return self._length
