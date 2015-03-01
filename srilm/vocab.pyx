"""
Module dealing with vocabulary
"""

from cython.operator cimport dereference as deref
from cpython cimport array

cdef class Vocab:
    """A Vocabulary manages a mapping between word string and word index"""
    def __cinit__(self, bint unk_is_word = True):
        self.thisptr = new c_vocab.Vocab()
        cdef Boolean *b = &self.thisptr.unkIsWord()
        b[0] = unk_is_word # very important

    def __dealloc__(self):
        del self.thisptr

    property unk:
        """Index of the unknown word '<unk>'"""
        def __get__(self):
            return self.thisptr.unkIndex()

    property bos:
        """Index of the beginning of sentence word '<s>'"""
        def __get__(self):
            return self.thisptr.ssIndex()

    property eos:
        """Index of the end of sentence word '</s>'"""
        def __get__(self):
            return self.thisptr.seIndex()

    property pau:
        """Index of the pause-filler word '-pau-'"""
        def __get__(self):
            return self.thisptr.pauseIndex()

    def add(self, VocabString token):
        """Add a word with an interned index"""
        return self.thisptr.addWord(token)

    def get(self, key):
        """Get the index of a string or the string of an index"""
        return self.__getitem__(key)

    def remove(self, key):
        """Remove a word by string or index"""
        self.__delitem__(key)

    def read(self, const char *fname):
        """Read vocabulary from a file"""
        cdef File *fptr
        fptr = new File(fname, 'r', 0)
        cdef bint ok = self.thisptr.read(deref(fptr))
        del fptr
        return ok

    def write(self, const char *fname):
        """Write vocabulary to a file"""
        cdef File *fptr
        fptr = new File(fname, 'w', 0)
        self.thisptr.write(deref(fptr))
        del fptr

    def index(self, words):
        """Map a list of word strings to an array of word indices"""
        from array import array
        res = array('I', [])
        cdef VocabIndex index
        for w in words:
            index = self.thisptr.getIndex(<VocabString>w, self.thisptr.unkIndex())
            res.append(index)
        return res

    def string(self, index):
        """Map an array of word indices to a list of word strings"""
        cdef VocabString word
        res = []
        cdef VocabIndex i
        for i in index:
            word = self.thisptr.getWord(i)
            if word == NULL:
                raise IndexError('Out of vocabulary index')
            res.append(word)
        return res

    def __iter__(self):
        self.iterptr = new c_vocab.VocabIter(deref(self.thisptr))
        if self.iterptr == NULL:
            raise MemoryError
        return self

    def __next__(self):
        """Iterator returns a tuple of (string, index)"""
        cdef VocabIndex index = 0
        cdef VocabString s = self.iterptr.next(index)
        if s == NULL:
            del self.iterptr
            raise StopIteration
        else:
            return (<bytes>s, index)

    def __contains__(self, VocabString token):
        return self.thisptr.getIndex(token) != c_vocab.Vocab_None

    def __getitem__(self, key):
        """Get the index of a string or the string of an index"""
        cdef VocabString word
        cdef VocabIndex index
        if isinstance(key, basestring):
            index = self.thisptr.getIndex(<VocabString>key)
            return None if index == c_vocab.Vocab_None else index
        elif isinstance(key, (int, long)):
            word = self.thisptr.getWord(key)
            return None if word == NULL else <bytes>word
        else:
            raise TypeError('Expect string or int')

    def __delitem__(self, key):
        """Remove a word by string or index"""
        if isinstance(key, basestring):
            self.thisptr.remove(<VocabString>key)
        elif isinstance(key, (int, long)):
            self.thisptr.remove(<VocabIndex>key)
        else:
            raise TypeError('Expect string or int')

    def __len__(self):
        """Get the number of words in the vocabulary"""
        return self.thisptr.numWords()
