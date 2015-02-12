from cython.operator cimport dereference as deref

VOCAB_NONE = Vocab_None

cdef class vocab:

    def __cinit__(self, VocabIndex start = 0, VocabIndex end = Vocab_None-1):
        self.thisptr = new Vocab(start, end)

    def __dealloc__(self):
        del self.thisptr

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

    def __iter__(self):
        self.iterptr = new VocabIter(deref(self.thisptr))
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
        return self.thisptr.getIndex(token) != Vocab_None

    def __getitem__(self, key):
        cdef VocabString word
        cdef VocabIndex index
        if isinstance(key, basestring):
            index = self.thisptr.getIndex(<VocabString>key)
            return None if index == Vocab_None else index
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
