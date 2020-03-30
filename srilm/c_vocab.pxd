from libcpp cimport bool
from srilm.common cimport File

cdef extern from "Boolean.h":
    ctypedef bool Boolean

cdef extern from "Vocab.h":
    ctypedef unsigned int VocabIndex
    ctypedef const char* VocabString
    cdef VocabIndex Vocab_None
    cdef cppclass Vocab:
        Vocab() # use default
        VocabIndex addWord(VocabString token)
        VocabString getWord(VocabIndex index)
        VocabIndex getIndex(VocabString token, VocabIndex unkIndex)
        VocabIndex getIndex(VocabString token)
        void remove(VocabString token)
        void remove(VocabIndex index)
        unsigned int read(File &file)
        void write(File &file)
        unsigned int numWords()
        VocabIndex &unkIndex()
        VocabIndex &ssIndex()
        VocabIndex &seIndex()
        VocabIndex &pauseIndex()
        Boolean &unkIsWord()
        unsigned int length(const VocabIndex *words)

    cdef cppclass VocabIter:
        VocabIter(Vocab &vocab)
        void init()
        VocabString next()
        VocabString next(VocabIndex &index)

cdef extern from "SubVocab.h":
    cdef cppclass SubVocab:
        SubVocab(Vocab &baseVocab, Boolean keepNonwords)
        VocabIndex addWord(VocabIndex wid)
