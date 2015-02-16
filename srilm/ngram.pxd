cimport c_vocab
from c_vocab cimport VocabIndex, VocabString
from vocab cimport Vocab
from common cimport File, LogP, Boolean, TextStats, LogPtoPPL, LogP2, Prob
cimport c_discount

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats:
        NgramStats(c_vocab.Vocab &vocab, unsigned int order)
        unsigned getorder()
        NgramCount *findCount(const VocabIndex *words)
        NgramCount *insertCount(const VocabIndex *words)
        Boolean removeCount(const VocabIndex *words, NgramCount *removedData)
        Boolean read(File &file)
        Boolean readBinary(File &file)
        void write(File &file)
        void writeBinary(File &file)
        unsigned int countFile(File &file)
        unsigned int countSentence(const VocabString *words) # only this one respects addSentSent and addSentEnd!!!
        NgramCount sumCounts()
        Boolean openVocab
        Boolean addSentStart
        Boolean addSentEnd
        c_vocab.Vocab &vocab

    cdef cppclass NgramsIter:
        NgramsIter(NgramStats &ngrams, VocabIndex *keys, unsigned order, int(*sort)(VocabIndex, VocabIndex))
        NgramCount *next()

cdef class Stats:
    cdef NgramStats *thisptr
    cdef VocabIndex *keysptr
    cdef Vocab _vocab

cdef class StatsIter:
    cdef NgramsIter *iterptr
    cdef VocabIndex *keysptr
    cdef unsigned int _iter_order

cdef extern from "Ngram.h":
    cdef const unsigned defaultNgramOrder
    cdef cppclass BOnode:
        BOnode()

    cdef cppclass Ngram:
        Ngram(c_vocab.Vocab &vocab, unsigned order)
        unsigned setorder(unsigned neworder)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        LogP countsProb(NgramStats &counts, TextStats &stats, unsigned order)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats, const char *escapeString, Boolean entropy)
        Boolean estimate(NgramStats &stats, c_discount.Discount **discounts)
        Boolean estimate(NgramStats &stats, NgramCount *mincounts, NgramCount *maxcounts)
        unsigned pplFile(File &file, TextStats &stats, const char *escapeString)
        NgramCount numNgrams(unsigned int n) const

    cdef cppclass NgramBOsIter:
        NgramBOsIter(const Ngram &lm, VocabIndex *keys, unsigned order, int(*sort)(VocabIndex, VocabIndex))
        BOnode *next()

    cdef cppclass NgramProbsIter:
        NgramProbsIter(const BOnode &bonode, int(*sort)(VocabIndex, VocabIndex))
        LogP *next(VocabIndex &word)

cdef class Lm:
    cdef Ngram *thisptr
    cdef VocabIndex *keysptr
    cdef c_discount.Discount **dlistptr
    cdef Vocab _vocab
    cdef list _dlist

cdef class LmIterContext:
    cdef NgramBOsIter *iterptr
    cdef VocabIndex *keysptr
    cdef unsigned int _iter_order

cdef class LmIterProb:
    cdef NgramProbsIter *iterptr
