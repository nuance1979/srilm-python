from srilm.common cimport Boolean, LogP, File
from srilm.c_vocab cimport VocabIndex, Vocab_None
from srilm cimport c_vocab
from srilm.stats cimport NgramStats
from srilm.ngram cimport Ngram
from srilm.vocab cimport Vocab
from srilm cimport base

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

