cimport c_vocab
from c_vocab cimport VocabIndex, VocabString
from vocab cimport Vocab
from common cimport File, LogP, Boolean, TextStats, LogPtoPPL, LogP2, Prob
cimport common

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats:
        NgramStats(c_vocab.Vocab &vocab, unsigned int order)
        unsigned getorder()
        NgramCount *findCount(const VocabIndex *words)
        NgramCount *insertCount(const VocabIndex *words)
        Boolean removeCount(const VocabIndex *words, NgramCount *removedData)
        Boolean read(File &file)
        void write(File &file)
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
    cdef NgramsIter *iterptr
    cdef VocabIndex *keysptr
    cdef Vocab _vocab

cdef extern from "Ngram.h":
    cdef const unsigned defaultNgramOrder
    cdef cppclass Ngram:
        Ngram(c_vocab.Vocab &vocab, unsigned order)
        unsigned setorder(unsigned neworder)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        LogP countsProb(NgramStats &counts, TextStats &stats, unsigned order)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats, const char *escapeString, Boolean entropy)
        Boolean estimate(NgramStats &stats, common.Discount **discounts)
        Boolean estimate(NgramStats &stats, NgramCount *mincounts, NgramCount *maxcounts)
        unsigned pplFile(File &file, TextStats &stats, const char *escapeString)
        NgramCount numNgrams(unsigned int n) const

cdef class Lm:
    cdef Ngram *thisptr
    cdef VocabIndex *keysptr
