cimport c_vocab
from c_vocab cimport SubVocab
from stats cimport NgramStats
from common cimport File, Boolean

cdef extern from "stdlib.h":
    cdef void srand48(long seed) # need to declare because it's not in libc.stdlib

cdef extern from "Prob.h":
    ctypedef float LogP
    ctypedef double LogP2
    ctypedef double Prob
    cdef Prob LogPtoProb(LogP2 prob)
    cdef LogP ProbToLogP(Prob prob)
    cdef LogP2 MixLogP(LogP2 prob1, LogP2 prob2, double lambda0)
    cdef LogP2 AddLogP(LogP2 x, LogP2 y)
    cdef LogP2 SubLogP(LogP2 x, LogP2 y)
    cdef LogP2 weightLogP(double weight, LogP2 prob)

cdef extern from "../../lm/src/ngram-class.cc":
    cdef cppclass UniqueWordClasses:
        UniqueWordClasses(c_vocab.Vocab &v, SubVocab &classVocab)
        void initialize(NgramStats &counts, SubVocab &noclassVocab)
        void fullMerge(unsigned numClasses)
        void incrementalMerge(unsigned numClasses)
        Boolean readClasses(File &file)
        void writeClasses(File &file)
        void writeCounts(File &file) # write class ngram counts
