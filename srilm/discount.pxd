from srilm cimport c_discount
from srilm.c_discount cimport ModKneserNey, KneserNey, GoodTuring, WittenBell, ConstDiscount, AddSmooth, NaturalDiscount
from srilm.stats cimport Stats

cdef class Discount:
    cdef c_discount.Discount *thisptr
    cdef void _init_thisptr(self)
    cdef void _get_discount(self)
    cdef public method, discount, interpolate, min_count, max_count
