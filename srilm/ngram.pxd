from vocab cimport Vocab, VocabIndex
from srilm_file cimport File
from prob cimport LogP
from boolean cimport Boolean

cdef extern from "Ngram.h":
    cdef const unsigned defaultNgramOrder
    cdef cppclass Ngram:
        Ngram(Vocab &vocab, unsigned order)
        unsigned setorder(unsigned neworder)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)

cdef class ngram:
    cdef Ngram *thisptr

cdef extern from "NgramStats.h":
    cdef cppclass NgramStats:
        NgramStats(Vocab &vocab, unsigned int order)
        unsigned getorder()

cdef class stats:
    cdef NgramStats *thisptr
