from srilm.common cimport File, Boolean
from srilm cimport c_vocab
from srilm.c_vocab cimport VocabIndex, VocabString, VocabIter

cdef class Vocab:
    cdef c_vocab.Vocab *thisptr
    cdef VocabIter *iterptr
