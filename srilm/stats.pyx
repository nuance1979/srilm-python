"""
Module for dealing with ngram counts
"""

from cython.operator cimport dereference as deref
from srilm cimport c_vocab
from srilm.c_vocab cimport Vocab_None, VocabIndex, VocabString
from srilm.vocab cimport Vocab
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from srilm.common cimport _fill_buffer_with_array, _create_array_from_buffer
from srilm.stats cimport NgramStats, NgramCount, NgramsIter
from srilm.common cimport File, Boolean


cdef class Stats:
    """Ngram counts stored in a trie"""
    cdef NgramStats *thisptr
    cdef VocabIndex *keysptr
    cdef Vocab _vocab

    def __cinit__(self, Vocab v, unsigned int order, open_vocab=False, add_bos=True, add_eos=True):
        self.thisptr = new NgramStats(deref(<c_vocab.Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order + 1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError
        self.thisptr.openVocab = open_vocab  # very important and easy to miss!!!
        self.thisptr.addSentStart = add_bos  # turn it on explicitly
        self.thisptr.addSentEnd = add_eos  # ditto
        self._vocab = v  # keep a python reference to vocab

    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.thisptr

    property order:
        """Ngram order of the counts"""
        def __get__(self):
            return self.thisptr.getorder()

    def add(self, words, NgramCount inc):
        """Increase the count of an ngram"""
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] += inc
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] += inc

    def remove(self, words):
        """Remove an ngram"""
        cdef NgramCount count
        cdef Boolean b
        if not words:
            b = self.thisptr.removeCount(NULL, &count)
            return count if b else 0
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            b = self.thisptr.removeCount(self.keysptr, &count)
            return count if b else 0

    def read(self, fname, binary=False):
        """Read counts from a file"""
        mode = b'rb' if binary else b'r'
        cdef File *fptr = new File(fname.encode(), mode, 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        ok = self.thisptr.read(deref(fptr))
        del fptr
        return ok

    def write(self, fname, binary=False):
        """Write counts to a file"""
        mode = b'wb' if binary else b'w'
        cdef File *fptr = new File(fname.encode(), mode, 0)
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
        """Count an array of indices"""
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
        cdef Py_ssize_t num_words = len(words)
        while j <= num_words:
            self.add(words[i:j], 1)
            i += 1
            j += 1
        return num_words

    def count_string(self, string):
        """Count a list of strings"""
        words = string.encode().split()
        cdef Py_ssize_t slen = len(words)
        cdef VocabString *buff = <VocabString *>PyMem_Malloc((slen + 1) * sizeof(VocabString))
        if buff == NULL:
            raise MemoryError
        cdef Py_ssize_t i
        for i in range(slen):
            buff[i] = words[i]
        buff[slen] = NULL  # another different ending convention!!!
        cdef unsigned int c = self.thisptr.countSentence(buff)
        PyMem_Free(buff)
        return c

    def count_file(self, fname):
        """Count a text file"""
        cdef File *fptr = new File(fname.encode(), b'r', 0)
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
            for w, c in self.iter(i + 1):
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
        """Get the count of an ngram"""
        cdef NgramCount *p
        if not words:
            p = self.thisptr.findCount(NULL)
            return 0 if p == NULL else deref(p)
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.findCount(self.keysptr)
            return 0 if p == NULL else deref(p)

    def __setitem__(self, words, NgramCount count):
        """Set the count of an ngram"""
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] = count
        else:
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] = count

    def __delitem__(self, words):
        """Remove an ngram"""
        self.remove(words)

    def __len__(self):
        """Get the total count of ngrams of the highest order"""
        cdef NgramCount s = 0
        cdef NgramCount i
        for _, i in self:
            s += i
        return s

    def __iter__(self):
        """Iterate through ngrams of the highest order"""
        return self.iter(self.order)

    def iter(self, unsigned int order):
        """Iterate through ngrams of a certain order"""
        if order < 1 or order > self.order:
            raise ValueError('Invalid order')
        return _create_stats_iter(self.thisptr, order)


cdef StatsIter _create_stats_iter(NgramStats *statsptr, unsigned int order):
    it = StatsIter()
    it.keysptr = <VocabIndex *>PyMem_Malloc((order + 1) * sizeof(VocabIndex))  # iterator should manage its own buffer
    if it.keysptr == NULL:
        raise MemoryError
    it.iterptr = new NgramsIter(deref(statsptr), it.keysptr, order, NULL)
    if it.iterptr == NULL:
        raise MemoryError
    it._iter_order = order
    return it


cdef class StatsIter:
    """Ngram stats iterator"""
    cdef NgramsIter *iterptr
    cdef VocabIndex *keysptr
    cdef unsigned int _iter_order

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
