from c_vocab cimport VocabIndex
cimport c_vocab
from common cimport Boolean, TextStats, File, LogP

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount

cdef extern from "LM.h":
    cdef cppclass LM:
        LM(c_vocab.Vocab &vocab)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats)
        unsigned pplFile(File &file, TextStats &stats)
        VocabIndex generateWord(const VocabIndex *context)
        VocabIndex *generateSentence(unsigned maxWords, VocabIndex *sentence, VocabIndex *prefix)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        Boolean writeBinary(File &file)

cdef class Lm:
    cdef LM *lmptr
