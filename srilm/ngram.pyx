from cython.operator cimport dereference as deref
cimport c_vocab
from c_vocab cimport Vocab_None
from vocab cimport Vocab
from cpython cimport array
from array import array
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from discount cimport ModKneserNey, KneserNey, GoodTuring, WittenBell, Discount

cdef inline bint _is_indices(words):
    return isinstance(words, array) and words.typecode == 'I'

cdef inline void _fill_buffer_with_array(unsigned int order, VocabIndex *buff, array.array words):
    cdef int n = min(order, len(words))
    cdef int i
    for i in range(n):
        buff[i] = words[i]
    buff[n] = Vocab_None

cdef inline array.array _create_array_from_buffer(unsigned int order, VocabIndex *buff):
    cdef array.array a = array('I', [])
    cdef int i
    for i in range(order):
        a.append(buff[i])
    return a

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
        elif _is_indices(words):
            _fill_buffer_with_array(self.order, self.keysptr, words)
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
        elif _is_indices(words):
            _fill_buffer_with_array(self.order, self.keysptr, words)
            b = self.thisptr.removeCount(self.keysptr, &count)
            return count if b else 0
        else:
            raise TypeError('Expect array')

    def read(self, const char *fname, binary = False):
        mode = 'rb' if binary else 'r'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr.error():
            raise IOError
        ok = self.thisptr.read(deref(fptr))
        del fptr
        return ok

    def write(self, const char *fname, binary = False):
        mode = 'wb' if binary else 'w'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr.error():
            raise IOError
        if binary:
            self.thisptr.writeBinary(deref(fptr))
        else:
            self.thisptr.write(deref(fptr))
        del fptr

    def count(self, words):
        cdef int i = 0
        cdef int j = self.order
        cdef int l = len(words)
        if _is_indices(words):
            while j <= l:
                self.add(words[i:j], 1)
                i += 1
                j += 1
            return l
        else:
            raise TypeError('Expect array')

    def count_string(self, string):
        words = string.split()
        cdef int slen = len(words)
        cdef VocabString *buff = <VocabString *>PyMem_Malloc((slen+1) * sizeof(VocabString))
        if buff == NULL:
            raise MemoryError
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

    def sum(self):
        """Recompute lowerer order counts by summing higher order counts"""
        return self.thisptr.sumCounts()

    def make_test(self):
        """Return a Stats() object for LM testing by stripping away unnecessary counts"""
        cdef array.array w
        cdef NgramCount c
        cdef unsigned int i
        cdef Stats s = Stats(self._vocab, self.order)
        for w, c in self:
            s[w] = c
            if w[0] == self.thisptr.vocab.ssIndex():
                for i in range(2, self.order):
                    s[w[:i]] += c
        return s

    def __getitem__(self, words):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.findCount(NULL)
            return 0 if p == NULL else deref(p)
        elif _is_indices(words):
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.findCount(self.keysptr)
            return 0 if p == NULL else deref(p)
        else:
            raise TypeError('Expect array')

    def __setitem__(self, words, NgramCount count):
        cdef NgramCount *p
        if not words:
            p = self.thisptr.insertCount(NULL)
            p[0] = count
        elif _is_indices(words):
            _fill_buffer_with_array(self.order, self.keysptr, words)
            p = self.thisptr.insertCount(self.keysptr)
            p[0] = count
        else:
            raise TypeError('Expect array')

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

cdef class Lm:
    """Ngram language model"""
    def __cinit__(self, Vocab v, unsigned order = defaultNgramOrder):
        if order < 1:
            raise ValueError('Invalid order')
        self.thisptr = new Ngram(deref(<c_vocab.Vocab *>(v.thisptr)), order)
        if self.thisptr == NULL:
            raise MemoryError
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError
        self.dlistptr = <c_discount.Discount **>PyMem_Malloc(order * sizeof(c_discount.Discount *))
        if self.dlistptr == NULL:
            raise MemoryError
        cdef unsigned int i
        for i in range(order):
            self.dlistptr[i] = NULL
        self._vocab = v # keep a python reference to vocab
        self._dlist = []

    def __dealloc__(self):
        PyMem_Free(self.dlistptr)
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

    def prob(self, VocabIndex word, context):
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
        elif _is_indices(context):
            _fill_buffer_with_array(self.order-1, self.keysptr, context)
            return self.thisptr.wordProb(word, self.keysptr)
        else:
            raise TypeError('Expect array')

    def prob_ngram(self, ngram):
        """Return log probability of p(ngram[-1] | ngram[-2], ngram[-3], ...)

           Noe that this function takes ngram in its *natural* order.
        """
        cdef VocabIndex word = ngram[-1]
        cdef array.array context = ngram[:-1].reverse()
        return self.prob(word, context)

    def read(self, const char *fname, Boolean limitVocab = False):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr.error():
            raise IOError
        ok = self.thisptr.read(deref(fptr), limitVocab)
        del fptr
        return ok

    def write(self, const char *fname):
        cdef File *fptr = new File(fname, 'w', 0)
        if fptr.error():
            raise IOError
        self.thisptr.write(deref(fptr))
        del fptr

    def test(self, Stats ns):
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
            return (prob, denom, float('NaN'))

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
        cdef BOnode *p = self.iterptr.next();
        cdef array.array keys
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
