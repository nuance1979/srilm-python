cimport c_vocab
from c_vocab cimport VocabIndex, VocabString
from vocab cimport Vocab
from common cimport File, LogP, Boolean, TextStats
cimport c_discount
cimport base

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
        Boolean estimate(NgramStats &stats, c_discount.Discount **discounts)
        NgramCount numNgrams(unsigned int n) const
        void mixProbs(Ngram &lm2, double lambda0)

    cdef cppclass NgramBOsIter:
        NgramBOsIter(const Ngram &lm, VocabIndex *keys, unsigned order, int(*sort)(VocabIndex, VocabIndex))
        BOnode *next()

    cdef cppclass NgramProbsIter:
        NgramProbsIter(const BOnode &bonode, int(*sort)(VocabIndex, VocabIndex))
        LogP *next(VocabIndex &word)

cdef class Lm(base.Lm):
    cdef Ngram *thisptr
    cdef c_discount.Discount **dlistptr
    cdef list _dlist

cdef class LmIterContext:
    cdef NgramBOsIter *iterptr
    cdef VocabIndex *keysptr
    cdef unsigned int _iter_order

cdef class LmIterProb:
    cdef NgramProbsIter *iterptr

cdef extern from "NgramCountLM.h":
    cdef cppclass NgramCountLM:
        NgramCountLM(c_vocab.Vocab &vocab, unsigned order)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean estimate(NgramStats &stats)
        unsigned maxEMiters
        double minEMdelta

cdef class CountLm(base.Lm):
    cdef NgramCountLM *thisptr

cdef extern from "SubVocab.h":
    cdef cppclass SubVocab:
        SubVocab(c_vocab.Vocab &baseVocab, Boolean keepNonwords)
        VocabIndex addWord(VocabIndex wid)

cdef extern from "../../lm/src/ngram-class.cc":
    cdef cppclass UniqueWordClasses:
        UniqueWordClasses(c_vocab.Vocab &v, SubVocab &classVocab)
        void fullMerge(unsigned numClasses)
        void incrementalMerge(unsigned numClasses)
        Boolean readClasses(File &file)
        void writeClasses(File &file)

cdef extern from "SimpleClassNgram.h":
    cdef cppclass SimpleClassNgram:
        SimpleClassNgram(c_vocab.Vocab &vocab, SubVocab &classVocab, unsigned order)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean readClasses(File &file)

cdef class ClassLm(base.Lm):
    cdef SimpleClassNgram *thisptr

cdef extern from "CacheLM.h":
    cdef cppclass CacheLM:
        CacheLM(c_vocab.Vocab &vocab, unsigned historyLength)
        LogP wordProb(VocabIndex word, const VocabIndex *context)

cdef class CacheLm(base.Lm):
    cdef CacheLM *thisptr
    cdef unsigned _length
