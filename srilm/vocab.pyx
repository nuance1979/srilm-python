from cython.operator cimport dereference as deref
from cpython cimport array
from array import array

cdef class Vocab:

    def __cinit__(self, VocabIndex start = 0, VocabIndex end = c_vocab.Vocab_None-1):
        self.thisptr = new c_vocab.Vocab(start, end)

    def __dealloc__(self):
        del self.thisptr

    property unk:
        def __get__(self):
            return self.thisptr.unkIndex()

    def add(self, VocabString token):
        return self.thisptr.addWord(token)

    def get(self, key):
        return self.__getitem__(key)

    def remove(self, key):
        self.__delitem__(key)

    def read(self, const char *fname):
        cdef File *fptr
        fptr = new File(<const char*>fname, 'r', 1)
        cdef bint ok = self.thisptr.read(deref(fptr))
        del fptr
        return ok

    def write(self, const char *fname):
        cdef File *fptr
        fptr = new File(<const char*>fname, 'w', 1)
        self.thisptr.write(deref(fptr))
        del fptr

    def index(self, words):
        cdef array.array res = array('I', [])
        cdef VocabIndex index
        for w in words:
            index = self.thisptr.getIndex(<VocabString>w, self.thisptr.unkIndex())
            res.append(index)
        return res

    def string(self, index):
        cdef VocabString word
        res = []
        for i in index:
            word = self.thisptr.getWord(<VocabIndex>i)
            if word == NULL:
                raise IndexError('Out of vocabulary index')
            res.append(word)
        return res

    def __iter__(self):
        self.iterptr = new c_vocab.VocabIter(deref(self.thisptr))
        return self

    def __next__(self):
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
        if isinstance(key, basestring):
            self.thisptr.remove(<VocabString>key)
        elif isinstance(key, (int, long)):
            self.thisptr.remove(<VocabIndex>key)
        else:
            raise TypeError('Expect string or int')

    def __len__(self):
        return self.thisptr.numWords()
