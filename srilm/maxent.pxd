from common cimport Boolean, LogP, File
from c_vocab cimport VocabIndex, Vocab_None
cimport c_vocab
from ngram cimport NgramStats, Ngram
from vocab cimport Vocab

cdef extern from "MEModel.h":
    cdef cppclass MEModel:
        MEModel(c_vocab.Vocab &vocab, unsigned order)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        Ngram *getNgamLM()
        Boolean estimate(NgramStats &stats, double alpha, double sigma2)
        Boolean adapt(NgramStats &stats, double alpha, double sigma2)
        void setMaxIterations(unsigned max)

cdef class Lm:
    cdef MEModel *thisptr
    cdef VocabIndex *keysptr
    cdef Vocab _vocab
    cdef unsigned int _order

