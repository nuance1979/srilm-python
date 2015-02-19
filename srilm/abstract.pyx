from cython.operator cimport dereference as deref
from c_vocab cimport VocabIndex
from cpython cimport array
from ngram cimport Stats
from common cimport Boolean, File, LogP, Prob, LogP2, TextStats, LogPtoPPL
from vocab cimport Vocab

cdef tuple _compute_ppl(TextStats *tsptr):
    cdef LogP2 prob = tsptr.prob
    cdef double denom = tsptr.numWords - tsptr.numOOVs - tsptr.zeroProbs + tsptr.numSentences
    cdef Prob ppl = LogPtoPPL(prob / denom) if denom > 0 else float('NaN')
    return (prob, denom, ppl)

cdef class Lm:
    """Abstract class to encourage uniform interface"""
    def prob(self, VocabIndex word, context):
        raise NotImplementedError('Abstract class method')

    def prob_ngram(self, ngram):
        """Return log probability of p(ngram[-1] | ngram[-2], ngram[-3], ...)

           Noe that this function takes ngram in its *natural* order.
        """
        cdef VocabIndex word = ngram[-1]
        cdef array.array context = ngram[:-1].reverse()
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
