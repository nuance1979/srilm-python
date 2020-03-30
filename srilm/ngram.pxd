from srilm cimport c_vocab
from srilm.c_vocab cimport VocabIndex, VocabString, SubVocab
from srilm.vocab cimport Vocab
from srilm.stats cimport NgramStats, NgramCount
from srilm.common cimport File, LogP, Boolean, TextStats
from srilm cimport c_discount
from srilm cimport base

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
        void pruneProbs(double threshold, unsigned minorder, base.LM *historyLM)

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

cdef extern from "SimpleClassNgram.h":
    cdef cppclass SimpleClassNgram:
        SimpleClassNgram(c_vocab.Vocab &vocab, SubVocab &classVocab, unsigned order)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean readClasses(File &file)
        void writeClasses(File &file)

cdef class SimpleClassLm(base.Lm):
    cdef SimpleClassNgram *thisptr
    cdef SubVocab *_class_vocab_ptr

cdef extern from "CacheLM.h":
    cdef cppclass CacheLM:
        CacheLM(c_vocab.Vocab &vocab, unsigned historyLength)
        LogP wordProb(VocabIndex word, const VocabIndex *context)

cdef class CacheLm(base.Lm):
    cdef CacheLM *thisptr
    cdef unsigned _length
