cdef extern from "File.h":
    cdef cppclass File:
        File(const char *name, const char *mode, int exitOnError)
        
