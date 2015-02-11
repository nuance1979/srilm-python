from common cimport File

cdef extern from "Vocab.h":
    ctypedef unsigned int VocabIndex
    ctypedef const char* VocabString
    cdef VocabIndex Vocab_None
    cdef cppclass Vocab:
        Vocab(VocabIndex start, VocabIndex end)
        VocabIndex addWord(VocabString token)
        VocabString getWord(VocabIndex index)
        VocabIndex getIndex(VocabString token)
        void remove(VocabString token)
        void remove(VocabIndex index)
        unsigned int read(File &file)
        void write(File &file)
        unsigned int numWords()
        VocabIndex highIndex()

    cdef cppclass VocabIter:
        VocabIter(Vocab &vocab)
        void init()
        VocabString next()

cdef class vocab:
    cdef Vocab *thisptr
    cdef VocabIter *iterptr
