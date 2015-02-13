from vocab cimport Vocab, VocabIndex
from common cimport File, LogP, Boolean, TextStats, Discount, LogPtoPPL, LogP2, Prob

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats:
        NgramStats(Vocab &vocab, unsigned int order)
        unsigned getorder()
        NgramCount *findCount(const VocabIndex *words)
        NgramCount *insertCount(const VocabIndex *words)
        Boolean removeCount(const VocabIndex *words, NgramCount *removedData)
        Boolean read(File &file)
        void write(File &file)
        unsigned int countFile(File &file)
        unsigned int countString(char *sentence)
        NgramCount sumCounts()
        Boolean openVocab
        Boolean addSentStart
        Boolean addSentEnd

    cdef cppclass NgramsIter:
        NgramsIter(NgramStats &ngrams, VocabIndex *keys, unsigned order, int(*sort)(VocabIndex, VocabIndex))
        NgramCount *next()

cdef class stats:
    cdef NgramStats *thisptr
    cdef NgramsIter *iterptr
    cdef VocabIndex *keysptr

cdef extern from "Ngram.h":
    cdef const unsigned defaultNgramOrder
    cdef cppclass Ngram:
        Ngram(Vocab &vocab, unsigned order)
        unsigned setorder(unsigned neworder)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        LogP countsProb(NgramStats &counts, TextStats &stats, unsigned order)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats, const char *escapeString, Boolean entropy)
        Boolean estimate(NgramStats &stats, Discount **discounts)
        Boolean estimate(NgramStats &stats, NgramCount *mincounts, NgramCount *maxcounts)
        unsigned pplFile(File &file, TextStats &stats, const char *escapeString)
        NgramCount numNgrams(unsigned int n) const

cdef class lm:
    cdef Ngram *thisptr
    cdef VocabIndex *keysptr
