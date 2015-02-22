from common cimport Boolean, LogP, File
from c_vocab cimport VocabIndex, Vocab_None
cimport c_vocab
from stats cimport NgramStats
from ngram cimport Ngram
from vocab cimport Vocab
cimport base

cdef extern from "MEModel.h":
    cdef cppclass MEModel:
        MEModel(c_vocab.Vocab &vocab, unsigned order)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        Ngram *getNgramLM()
        Boolean estimate(NgramStats &stats, double alpha, double sigma2)
        Boolean adapt(NgramStats &stats, double alpha, double sigma2)
        void setMaxIterations(unsigned max)

cdef class Lm(base.Lm):
    cdef MEModel *thisptr

