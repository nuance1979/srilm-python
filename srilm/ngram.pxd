from vocab cimport Vocab, VocabIndex
from common cimport File, LogP, Boolean, TextStats

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats:
        NgramStats(Vocab &vocab, unsigned int order)
        unsigned getorder()

cdef class stats:
    cdef NgramStats *thisptr

cdef extern from "Ngram.h":
    cdef const unsigned defaultNgramOrder
    cdef cppclass Ngram:
        Ngram(Vocab &vocab, unsigned order)
        unsigned setorder(unsigned neworder)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats, const char *escapeString, Boolean entropy)
        unsigned pplFile(File &file, TextStats &stats, const char *escapeString)

cdef class ngram:
    cdef Ngram *thisptr

