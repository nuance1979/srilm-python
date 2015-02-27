from c_vocab cimport VocabIndex
cimport c_vocab
from common cimport Boolean, TextStats, File, LogP
from vocab cimport Vocab

cdef extern from "NgramStats.h":
    ctypedef unsigned long NgramCount
    cdef cppclass NgramStats

cdef extern from "LM.h":
    cdef cppclass LM:
        LM(c_vocab.Vocab &vocab)
        LogP wordProb(VocabIndex word, const VocabIndex *context)
        LogP countsProb(NgramStats &counts, TextStats &stats, unsigned order, Boolean entropy)
        NgramCount pplCountsFile(File &file, unsigned order, TextStats &stats)
        unsigned pplFile(File &file, TextStats &stats)
        VocabIndex generateWord(const VocabIndex *context)
        VocabIndex *generateSentence(unsigned maxWords, VocabIndex *sentence, VocabIndex *prefix)
        Boolean read(File &file, Boolean limitVocab)
        Boolean write(File &file)
        Boolean writeBinary(File &file)
        void debugme(unsigned level)
        unsigned debuglevel() const
        Boolean running() const
        Boolean running(Boolean newstate)
        unsigned probServer(unsigned port, unsigned maxClients)

cdef class Lm:
    cdef LM *lmptr
    cdef VocabIndex *keysptr
    cdef Vocab _vocab
    cdef unsigned int _order

cdef extern from "LMClient.h":
    cdef cppclass LMClient:
        LMClient(c_vocab.Vocab &vocab, const char *server, unsigned order, unsigned cacheOrder)

cdef class ClientLm(Lm):
    cdef LMClient *thisptr
