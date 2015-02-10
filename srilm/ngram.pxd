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
