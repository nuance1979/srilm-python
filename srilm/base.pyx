from cython.operator cimport dereference as deref
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free
from c_vocab cimport VocabIndex, Vocab_None
from stats cimport Stats
from common cimport Boolean, File, LogP, Prob, LogP2, TextStats, LogPtoPPL, _fill_buffer_with_array, _create_array_from_buffer
from vocab cimport Vocab
from array import array

cdef tuple _compute_ppl(TextStats *tsptr):
    cdef LogP2 prob = tsptr.prob
    cdef double denom = tsptr.numWords - tsptr.numOOVs - tsptr.zeroProbs + tsptr.numSentences
    cdef Prob ppl = LogPtoPPL(prob / denom) if denom > 0 else float('NaN')
    return (prob, denom, ppl)

cdef class Lm:
    """Base class to share common code and to encourage uniform interface"""
    def __cinit__(self, Vocab v, unsigned order):
        self.keysptr = <VocabIndex *>PyMem_Malloc((order+1) * sizeof(VocabIndex))
        if self.keysptr == NULL:
            raise MemoryError
        self._vocab = v
        self._order = order

    def __dealloc__(self):
        PyMem_Free(self.keysptr)

    property order:
        def __get__(self):
            return self._order

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
            return self.lmptr.wordProb(word, self.keysptr)
        else:
            _fill_buffer_with_array(self._order, self.keysptr, context)
            return self.lmptr.wordProb(word, self.keysptr)

    def prob_ngram(self, ngram):
        """Return log probability of p(ngram[-1] | ngram[-2], ngram[-3], ...)

           Noe that this function takes ngram in its *natural* order.
        """
        cdef VocabIndex word = ngram[-1]
        context = ngram[:-1].reverse()
        return self.prob(word, context)
    
    def read(self, const char *fname, Boolean limitVocab = False, binary = False):
        mode = 'rb' if binary else 'r'
        cdef File *fptr = new File(fname, mode, 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        ok = self.lmptr.read(deref(fptr), limitVocab)
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
            self.lmptr.writeBinary(deref(fptr))
        else:
            self.lmptr.write(deref(fptr))
        del fptr

    def train(self, Stats ts):
        raise NotImplementedError('Abstract class method')

    def test(self, Stats ns):
        cdef TextStats *tsptr = new TextStats()
        if tsptr == NULL:
            raise MemoryError
        cdef LogP p = self.lmptr.countsProb(deref(ns.thisptr), deref(tsptr), ns.order, False)
        prob, denom, ppl = _compute_ppl(tsptr)
        del tsptr
        return (prob, denom, ppl)

    def test_text_file(self, fname):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        cdef TextStats *tsptr = new TextStats()
        if tsptr == NULL:
            raise MemoryError
        cdef unsigned c = self.lmptr.pplFile(deref(fptr), deref(tsptr))
        prob, denom, ppl = _compute_ppl(tsptr)
        del tsptr
        return (prob, denom, ppl)

    def test_counts_file(self, fname, unsigned order):
        cdef File *fptr = new File(fname, 'r', 0)
        if fptr == NULL:
            raise MemoryError
        elif fptr.error():
            raise IOError
        cdef TextStats *tsptr = new TextStats()
        if tsptr == NULL:
            raise MemoryError
        cdef NgramCount c = self.lmptr.pplCountsFile(deref(fptr), order, deref(tsptr))
        prob, denom, ppl = _compute_ppl(tsptr)
        del tsptr
        return (prob, denom, ppl)

    property debug_level:

        def __get__(self):
            return self.lmptr.debuglevel()

        def __set__(self, unsigned level):
            self.lmptr.debugme(level)

    property running:

        def __get__(self):
            return self.lmptr.running()

        def __set__(self, Boolean newstate):
            self.lmptr.running(newstate)

    def rand_gen(self, unsigned max_word):
        cdef VocabIndex *sent = <VocabIndex *>PyMem_Malloc((max_word+1) * sizeof(VocabIndex))
        if sent == NULL:
            raise MemoryError
        sent[max_word] = Vocab_None # sentinel
        self.lmptr.generateSentence(max_word, sent, NULL)
        cdef unsigned sent_len = self._vocab.thisptr.length(sent)
        a = _create_array_from_buffer(sent_len, sent)
        res = self._vocab.string(a)
        PyMem_Free(sent)
        return res

