from libcpp cimport bool
from array import array
from cpython cimport array
cimport c_vocab

cdef extern from "Vocab.h":
    ctypedef unsigned int VocabIndex
    cdef VocabIndex Vocab_None

cdef extern from "Boolean.h":
    ctypedef bool Boolean

cdef extern from "Prob.h":
    ctypedef float LogP
    ctypedef double LogP2
    ctypedef double Prob
    Prob LogPtoPPL(LogP2 prob)

cdef extern from "File.h":
    cdef cppclass File:
        File(const char *name, const char *mode, int exitOnError)
        Boolean error()

cdef extern from "Counts.h":
    ctypedef double FloatCount
        
cdef extern from "TextStats.h":
    cdef cppclass TextStats:
        TextStats()
        LogP2 prob
        FloatCount zeroProbs
        FloatCount numSentences
        FloatCount numWords
        FloatCount numOOVs

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats:
        NgramStats(c_vocab.Vocab &vocab, unsigned int order)

cdef inline void _fill_buffer_with_array(unsigned int order, VocabIndex *buff, array.array words):
    cdef int n = min(order, len(words))
    cdef int i
    for i in range(n):
        buff[i] = words[i]
    buff[n] = Vocab_None

cdef inline array.array _create_array_from_buffer(unsigned int order, VocabIndex *buff):
    cdef array.array a = array('I', [])
    cdef int i
    for i in range(order):
        a.append(buff[i])
    return a
