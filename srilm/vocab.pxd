from common cimport File, Boolean
cimport c_vocab
from c_vocab cimport VocabIndex, VocabString, VocabIter

cdef class Vocab:
    cdef c_vocab.Vocab *thisptr
    cdef VocabIter *iterptr
