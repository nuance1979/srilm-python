from srilm cimport c_vocab
from srilm.c_vocab cimport VocabIndex, VocabString
from srilm.vocab cimport Vocab
from srilm.common cimport File, Boolean

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
        void write(File &file, unsigned int order)
        void writeBinary(File &file, unsigned order)
        unsigned int countFile(File &file)
        unsigned int countSentence(const VocabString *words) # only this one respects addSentSent and addSentEnd!!!
        NgramCount sumCounts(unsigned int order)
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
