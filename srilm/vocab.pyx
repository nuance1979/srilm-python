from cython.operator cimport dereference as deref

cdef class vocab:
    vocab_none = Vocab_None

    def __cinit__(self, VocabIndex start = 0, VocabIndex end = Vocab_None-1):
        self.thisptr = new Vocab(start, end)

    def __dealloc__(self):
        del self.thisptr

    def addWord(self, token):
        return self.thisptr.addWord(<VocabString>token)

    def getWord(self, VocabIndex index):
        return <bytes>self.thisptr.getWord(index)

    def getIndex(self, token):
        return self.thisptr.getIndex(<VocabString>token)

    def remove(self, token):
        self.thisptr.remove(<VocabString>token)

    def remove(self, VocabIndex index):
        self.thisptr.remove(index)

    def numWords(self):
        return self.thisptr.numWords()

    def highIndex(self):
        return self.thisptr.highIndex()

    def read(self, fname):
        cdef File *fptr
        cdef unsigned int i
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'r', 1)
        i = self.thisptr.read(deref(fptr))
        del fptr
        return i

    def write(self, fname):
        cdef File *fptr
        if not isinstance(fname, bytes):
            raise TypeError('Expect string')
        fptr = new File(<const char*>fname, 'w', 1)
        self.thisptr.write(deref(fptr))
        del fptr

    def __iter__(self):
        self.iterptr = new VocabIter(deref(self.thisptr))
        return self

    def __next__(self):
        cdef VocabString s = self.iterptr.next()
        if s == NULL:
            del self.iterptr
            raise StopIteration
        else:
            return <bytes>s

    def __contains__(self, token):
        return self.thisptr.getIndex(<VocabString>token) != Vocab_None

    def __getitem__(self, key):
        cdef VocabString word
        cdef VocabIndex i
        if isinstance(key, bytes):
            i = self.thisptr.getIndex(<VocabString>key)
            return None if i == Vocab_None else i
        elif isinstance(key, int):
            word = self.thisptr.getWord(key)
            return None if word == NULL else <bytes>word
        else:
            raise TypeError('Expect string or int')

    def __delitem__(self, key):
        if isinstance(key, bytes):
            self.thisptr.remove(<bytes>key)
        elif isinstance(key, int):
            self.thisptr.remove(<VocabIndex>key)
        else:
            raise TypeError('Expect string or int')

    def __len__(self):
        return self.thisptr.numWords()
