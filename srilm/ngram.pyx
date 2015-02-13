from cython.operator cimport dereference as deref
cimport c_vocab
from c_vocab cimport Vocab_None
from vocab cimport Vocab
from cpython cimport array
from array import array
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from common cimport ModKneserNey, KneserNey, GoodTuring, WittenBell

cdef inline bint _isindices(words):
    return isinstance(words, array) and words.typecode == 'I'

cdef inline void _tobuffer(unsigned int order, VocabIndex *buff, array.array words):
    cdef int n = min(order, len(words))
    cdef int i
    for i in range(n):
        buff[i] = words[i]
    buff[n] = Vocab_None

cdef inline array.array _toarray(unsigned int order, VocabIndex *buff):
    cdef array.array a = array('I', [])
    cdef int i
    for i in range(order):
        a.append(buff[i])
    return a

cdef class Stats:
    """Holds ngram counts as a trie"""
    def __cinit__(self, Vocab v, unsigned int order):
        self.thisptr = new NgramStats(deref(<c_vocab.Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError
        self.thisptr.openVocab = False # very important and easy to miss!!!
        self.thisptr.addSentStart = True # turn it on explicitly
        self.thisptr.addSentEnd = True # ditto

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
        elif _isindices(words):
            _tobuffer(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] += inc
        else:
            raise TypeError('Expect array')

    def remove(self, words):
        cdef NgramCount count
        cdef Boolean b
        if not words:
            b = self.thisptr.removeCount(NULL, &count)
            return count if b else 0
        elif _isindices(words):
            _tobuffer(self.order, self.keysptr, words)
            b = self.thisptr.removeCount(self.keysptr, &count)
            return count if b else 0
        else:
            raise TypeError('Expect array')

    def read(self, const char *fname):
        cdef File *fptr = new File(fname, 'r', 0)
        if deref(fptr).error():
            raise IOError
        cdef bint ok = self.thisptr.read(deref(fptr))
        del fptr
        return ok

    def write(self, const char *fname):
        cdef File *fptr = new File(fname, 'w', 0)
        if deref(fptr).error():
            raise IOError
        self.thisptr.write(deref(fptr))
        del fptr

    def count(self, words):
        cdef int i = 0
        cdef int j = self.order
        cdef int l = len(words)
        if _isindices(words):
            while j <= l:
                self.add(words[i:j], 1)
                i += 1
                j += 1
            return l
        else:
            raise TypeError('Expect array')

    def count_string(self, const char *string):
        return self.thisptr.countString(<char*>string)

    def count_file(self, const char *fname):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr.error():
            raise IOError
        cdef unsigned int c = self.thisptr.countFile(deref(fptr))
        del fptr
        return c

    def sum(self):
        """Compute lowerer order ngram counts by summing higher order ngrams

           Note that Ngram LM training needs ngram counts of *all* orders but evaluation needs only as much as you need.
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
           you can get is by first calling count_string('this is a test') then sum().
           In contrast, if you need to evaluation on a sentence 'this is a test', you *only* need the following counts:
                             c('<s> this') = 1     # this is a 2-gram!
                             c('<s> this is') = 1
                             c('this is a') = 1
                             c('is a test') = 1
                             c('a test </s>') = 1
           No more no less.
        """
        return self.thisptr.sumCounts()

    def __getitem__(self, words):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.findCount(NULL)
            return 0 if p == NULL else deref(p)
        elif _isindices(words):
            _tobuffer(self.order, self.keysptr, words)
            p = self.thisptr.findCount(self.keysptr)
            return 0 if p == NULL else deref(p)
        else:
            raise TypeError('Expect array')

    def __setitem__(self, words, NgramCount count):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] = count
        elif _isindices(words):
            _tobuffer(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] = count
        else:
            raise TypeError('Expect array')

    def __delitem__(self, words):
        self.remove(words)

    def __iter__(self):
        self.iterptr = new NgramsIter(deref(self.thisptr), self.keysptr, self.order, NULL)
        return self

    def __next__(self):
        cdef NgramCount *p = self.iterptr.next()
        cdef array.array keys
        if p == NULL:
            del self.iterptr
            raise StopIteration
        else:
            keys = _toarray(self.order, self.keysptr)
            return (keys, deref(p))

    def __len__(self):
        cdef NgramCount s = 0
        cdef NgramCount i
        for _, i in self:
            s += i
        return s

cdef class Lm:
    """Ngram language model"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        self.thisptr = new Ngram(deref(<c_vocab.Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError

    def __dealloc__(self):
        PyMem_Free(self.keysptr)
        del self.thisptr

    property order:
        def __get__(self):
            return self.thisptr.setorder(0)

        def __set__(self, unsigned neworder):
            cdef VocabIndex *p = <VocabIndex *>PyMem_Realloc(self.keysptr, (neworder+1) * sizeof(VocabIndex))
            if p == NULL:
                raise MemoryError
            self.keysptr = p
            self.thisptr.setorder(neworder)

    def __len__(self):
        return self.thisptr.numNgrams(self.order)

    def _wordProb(self, VocabIndex word, context):
        """Return log probability of p(word | context)

           Note that the context is an ngram context in reverse order, i.e., if the text is
                      ... w_0 w_1 w_2 ...
           then this function computes
                      p(w_2 | w_1, w_0)
           and 'context' should be (w_1, w_0), *not* (w_0, w_1).
        """
        if not context:
            self.keysptr[0] = Vocab_None
            return self.thisptr.wordProb(word, self.keysptr)
        elif _isindices(context):
            _tobuffer(self.order-1, self.keysptr, context)
            return self.thisptr.wordProb(word, self.keysptr)
        else:
            raise TypeError('Expect array')

    def prob(self, ngram):
        """Return log probability of p(ngram[-1] | ngram[-2], ngram[-3], ...)

           Noe that this function takes ngram in its *natural* order.
        """
        cdef VocabIndex word = ngram[-1]
        cdef array.array context = ngram[:-1].reverse()
        return self._wordProb(word, context)

    def read(self, const char *fname, Boolean limitVocab = 0):
        cdef File *fptr = new File(fname, 'r', 0)
        if deref(fptr).error():
            raise IOError
        cdef bint ok = self.thisptr.read(deref(fptr), limitVocab)
        del fptr
        return ok

    def write(self, const char *fname):
        cdef File *fptr = new File(fname, 'w', 0)
        if deref(fptr).error():
            raise IOError
        self.thisptr.write(deref(fptr))
        del fptr

    def eval(self, Stats ns):
        cdef TextStats *tsptr = new TextStats()
        cdef LogP p = self.thisptr.countsProb(deref(ns.thisptr), deref(tsptr), ns.order)
        cdef double denom = tsptr.numWords - tsptr.numOOVs - tsptr.zeroProbs + tsptr.numSentences
        cdef LogP2 prob = tsptr.prob
        del tsptr
        cdef Prob ppl
        if denom > 0:
            ppl = LogPtoPPL(prob / denom)
            return (prob, denom, ppl)
        else:
            return (prob, denom, None)

    def train(self, Stats ts, smooth):
        cdef bint b
        cdef int i
        cdef Discount **discounts = <Discount **>PyMem_Malloc(self.order * sizeof(Discount *))
        if discounts == NULL:
            raise MemoryError
        for i in range(self.order):
            discounts[i] = <Discount *>new KneserNey()
            if discounts[i] == NULL:
                raise MemoryError
            discounts[i].interpolate = True
            b = discounts[i].estimate(deref(ts.thisptr), i+1)
            if not b:
                raise RuntimeError('error in discount estimator for order %d' % (i+1))
        b = self.thisptr.estimate(deref(ts.thisptr), discounts)
        for i in range(self.order):
            del discounts[i]
        PyMem_Free(discounts)
        return b
