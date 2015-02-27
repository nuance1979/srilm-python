"""
Module for dealing with ngram counts
"""

from cython.operator cimport dereference as deref
cimport c_vocab
from c_vocab cimport Vocab_None
from vocab cimport Vocab
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from array import array
from common cimport _fill_buffer_with_array, _create_array_from_buffer

cdef class Stats:
    """Holds ngram counts in a trie

    Note that Ngram LM training needs ngram counts of *all* orders but testing needs only as much as you need.
    For example, to train a 3-gram LM from a sentence 'this is a test', you need the following counts:
                             c('<s> this is') = 1
                             c('this is a') = 1
                             c('is a test') = 1
                             c('a test </s>') = 1
                             c('<s> this') = 1
                             c('this is') = 1
                             c('is a') = 1
                             c('a test') = 1
                             c('test </s>') = 1
                             c('<s>') = 1
                             c('this') = 1
                             c('is') = 1
                             c('a') = 1
                             c('test') = 1
                             c('</s>') = 1
    You can get it by calling count_string('this is a test').
    In contrast, if you need to testing on a sentence 'this is a test', you *only* need the following counts:
                             c('<s> this') = 1     # this is a 2-gram!
                             c('<s> this is') = 1
                             c('this is a') = 1
                             c('is a test') = 1
                             c('a test </s>') = 1
    No more no less. You can get it by calling make_test().
    """
    def __cinit__(self, Vocab v, unsigned int order, open_vocab = False, add_bos = True, add_eos = True):
        self.thisptr = new NgramStats(deref(<c_vocab.Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError
        self.thisptr.openVocab = open_vocab # very important and easy to miss!!!
        self.thisptr.addSentStart = add_bos # turn it on explicitly
        self.thisptr.addSentEnd = add_eos # ditto
        self._vocab = v # keep a python reference to vocab

    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.thisptr

    property order:
        def __get__(self):
            return self.thisptr.getorder()

    def add(self, words, NgramCount inc):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] += inc
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] += inc

    def remove(self, words):
        cdef NgramCount count
        cdef Boolean b
        if not words:
            b = self.thisptr.removeCount(NULL, &count)
            return count if b else 0
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            b = self.thisptr.removeCount(self.keysptr, &count)
            return count if b else 0

    def read(self, const char *fname, binary = False):
        mode = 'rb' if binary else 'r'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        ok = self.thisptr.read(deref(fptr))
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
            self.thisptr.writeBinary(deref(fptr), 0)
        else:
            self.thisptr.write(deref(fptr), 0)
        del fptr

    def count(self, words):
        if self.thisptr.addSentStart and words[0] != self._vocab.bos:
            words.insert(0, self._vocab.bos)
        if self.thisptr.addSentEnd and words[-1] != self._vocab.eos:
            words.append(self._vocab.eos)
        cdef Py_ssize_t i
        if words[0] != self._vocab.bos:
            self.add(words[0], 1)
        for i in range(2, self.order):
            self.add(words[:i], 1)
        i = 0
        cdef Py_ssize_t j = self.order
        cdef Py_ssize_t l = len(words)
        while j <= l:
            self.add(words[i:j], 1)
            i += 1
            j += 1
        return l

    def count_string(self, string):
        words = string.split()
        cdef Py_ssize_t slen = len(words)
        cdef VocabString *buff = <VocabString *>PyMem_Malloc((slen+1) * sizeof(VocabString))
        if buff == NULL:
            raise MemoryError
        cdef Py_ssize_t i
        for i in range(slen):
            buff[i] = words[i]
        buff[slen] = NULL # another different ending convention!!!
        cdef unsigned int c = self.thisptr.countSentence(buff)
        PyMem_Free(buff)
        return c

    def count_file(self, const char *fname):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr.error():
            raise IOError
        cdef unsigned int c = self.thisptr.countFile(deref(fptr))
        del fptr
        return c

    def sum_counts(self):
        """Recompute lowerer order counts by summing higher order counts"""
        return self.thisptr.sumCounts(self.order)

    def copy(self):
        """Return a copy of self"""
        cdef Stats s = Stats(self._vocab, self.order)
        cdef NgramCount c
        cdef unsigned int i
        for i in range(self.order):
            for w, c in self.iter(i+1):
                s[w] = c
        return s

    def make_test(self):
        """Prepare for testing by stripping away unnecessary counts"""
        cdef NgramCount c
        cdef unsigned int i
        cdef Stats s = Stats(self._vocab, self.order)
        for w, c in self:
            s[w] = c
            if w[0] == self.thisptr.vocab.ssIndex():
                for i in range(2, self.order):
                    s[w[:i]] += c
        # move data from s into self
        del self.thisptr
        self.thisptr = s.thisptr
        s.thisptr = NULL
        del s

    def __getitem__(self, words):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.findCount(NULL)
            return 0 if p == NULL else deref(p)
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.findCount(self.keysptr)
            return 0 if p == NULL else deref(p)

    def __setitem__(self, words, NgramCount count):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] = count
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] = count

    def __delitem__(self, words):
        self.remove(words)

    def __len__(self):
        cdef NgramCount s = 0
        cdef NgramCount i
        for _, i in self:
            s += i
        return s

    def __iter__(self):
        return self.iter(self.order)

    def iter(self, unsigned int order):
        if order < 1 or order > self.order:
            raise ValueError('Invalid order')        
        return _create_stats_iter(self.thisptr, order)

cdef StatsIter _create_stats_iter(NgramStats *statsptr, unsigned int order):
    it = StatsIter()
    it.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex)) # iterator should manage its own buffer
    if it.keysptr == NULL:
        raise MemoryError
    it.iterptr = new NgramsIter(deref(statsptr), it.keysptr, order, NULL)
    if it.iterptr == NULL:
        raise MemoryError
    it._iter_order = order
    return it
 
cdef class StatsIter:
    """Ngram stats iterator"""
    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.iterptr
    
    def __iter__(self):
        return self

    def __next__(self):
        cdef NgramCount *p = self.iterptr.next()
        if p == NULL:
            raise StopIteration
        else:
            keys = _create_array_from_buffer(self._iter_order, self.keysptr)
            return (keys, deref(p))
