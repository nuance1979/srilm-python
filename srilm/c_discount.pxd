from common cimport Boolean, NgramCount, NgramStats

cdef extern from "Discount.h":
    cdef cppclass Discount:
        Discount()
        Boolean estimate(NgramStats &counts, unsigned order)
        Boolean interpolate
        double discount(NgramCount count, NgramCount totalCount, NgramCount observedVocab)
        double lowerOrderWeight(NgramCount totalCount, NgramCount observedVocab, NgramCount min2Vocab, NgramCount min3Vocab)

    cdef cppclass ModKneserNey:
        ModKneserNey(unsigned mincount)
        double lowerOrderWeight(NgramCount totalCount, NgramCount observedVocab, NgramCount min2Vocab, NgramCount min3Vocab)

    cdef cppclass KneserNey:
        KneserNey(unsigned mincount)
        double lowerOrderWeight(NgramCount totalCount, NgramCount observedVocab, NgramCount min2Vocab, NgramCount min3Vocab)

    cdef cppclass GoodTuring:
        GoodTuring(unsigned mincount, unsigned maxcount)
        double discount(NgramCount count, NgramCount totalCount, NgramCount observedVocab)

    cdef cppclass WittenBell:
        WittenBell(double mincount)

    cdef cppclass ConstDiscount:
        ConstDiscount(double d, double mincount)
        double lowerOrderWeight(NgramCount totalCount, NgramCount observedVocab, NgramCount min2Vocab, NgramCount min3Vocab)

    cdef cppclass AddSmooth:
        AddSmooth(double delta, double mincount)
        double discount(NgramCount count, NgramCount totalCount, NgramCount observedVocab)

    cdef cppclass NaturalDiscount:
        NaturalDiscount(double mincount)
