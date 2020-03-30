"""
Module contains garden variety of ngram discounting methods
"""

from cython.operator cimport dereference as deref
from srilm cimport c_discount
from srilm.c_discount cimport ModKneserNey, KneserNey, GoodTuring, WittenBell, ConstDiscount, AddSmooth, NaturalDiscount
from srilm.stats cimport Stats

cdef class Discount:
    """Hold parameters for Ngram discount/smoothing method

    Note that you need one Discount() obj for each ngram order. If Discount.discount
    is provided, it will be used in Lm.train(); otherwise, it will be set with the
    estimated value.
    """
    cdef c_discount.Discount *thisptr
    cdef void _init_thisptr(self)
    cdef void _get_discount(self)
    cdef public method, discount, interpolate, min_count, max_count

    def __cinit__(self, method=None, discount=None, interpolate=None, min_count=None, max_count=None):
        self.method = None
        if method is not None:
            if method in ['kneser-ney', 'good-turing', 'witten-bell', 'chen-goodman', 'absolute', 'additive', 'natural']:
                self.method = method
            else:
                raise ValueError('Unknown smoothing method: %s' % method)
        self._set_default()
        if discount is not None:
            if self.method == 'good-turing':
                try:
                    self.discount = []
                    for d in discount:
                        assert isinstance(d, (int, long, float))
                        self.discount.append(d)
                except:
                    raise ValueError('Expect list of numbers for Good-Turing discount')
            elif not isinstance(discount, (int, long, float)):
                raise ValueError('Expect a number for discount')
            self.discount = discount
        if interpolate is not None:
            if not isinstance(interpolate, bool):
                raise ValueError('Expect a bool for interpolate')
            self.interpolate = interpolate
        if min_count is not None:
            if not isinstance(min_count, (int, long)):
                raise ValueError('Expect a number for min_count')
            self.min_count = min_count
        if max_count is not None:
            if not isinstance(max_count, (int, long)):
                raise ValueError('Expect a number for max_count')
            self.max_count = max_count
        self._init_thisptr()

    def __dealloc__(self):
        del self.thisptr

    def _set_default(self):
        """Set sensible default values for each type of discount"""
        self.interpolate = False
        self.min_count = 0
        self.max_count = float('Inf')
        if self.method == 'good-turing':
            self.min_count = 1
            self.max_count = 5
        elif self.method == 'additive':
            self.discount = 1.0
        elif self.method == 'absolute':
            self.discount = 0.5

    cdef void _init_thisptr(self):
        if self.method == 'kneser-ney':
            self.thisptr = <c_discount.Discount *>new KneserNey(self.min_count)
        elif self.method == 'good-turing':
            self.thisptr = <c_discount.Discount *>new GoodTuring(self.min_count, self.max_count)
        elif self.method == 'witten-bell':
            self.thisptr = <c_discount.Discount *>new WittenBell(self.min_count)
        elif self.method == 'chen-goodman':
            self.thisptr = <c_discount.Discount *>new ModKneserNey(self.min_count)
        elif self.method == 'absolute':
            self.thisptr = <c_discount.Discount *>new ConstDiscount(self.discount, self.min_count)
        elif self.method == 'additive':
            self.thisptr = <c_discount.Discount *>new AddSmooth(self.discount, self.min_count)
        elif self.method == 'natural':
            self.thisptr = <c_discount.Discount *>new NaturalDiscount(self.min_count)
        else:
            self.thisptr = new c_discount.Discount()
        if self.thisptr == NULL:
            raise MemoryError
        self.thisptr.interpolate = self.interpolate

    def estimate(self, Stats ts, unsigned int order):
        """Estimate the discount value from ngram counts Stats"""
        b = self.thisptr.estimate(deref(ts.thisptr), order)
        if b:
            self._get_discount()
        return b

    cdef void _get_discount(self):
        """Get estimated discount through 'reverse-engineering'"""
        if self.method == 'kneser-ney':
            self.discount = (<KneserNey *>self.thisptr).lowerOrderWeight(1, 1, 0, 0)
        elif self.method == 'good-turing':
            self.discount = []
            for i in range(self.min_count, self.max_count + 1):
                self.discount.append((<GoodTuring *>self.thisptr).discount(i, 0, 0))
        elif self.method == 'witten-bell':
            self.discount = None
        elif self.method == 'chen-goodman':
            self.discount = [(<ModKneserNey *>self.thisptr).lowerOrderWeight(1, 1, 0, 0), (<ModKneserNey *>self.thisptr).lowerOrderWeight(1, 1, 1, 0), (<ModKneserNey *>self.thisptr).lowerOrderWeight(1, 1, 1, 1)]

    def read(self, fname):
        """Read discount from a file"""
        try:
            import cPickle as pickle
        except:
            import pickle
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        self.method = data['method']
        self.discount = data['discount']
        self.interpolate = data['interpolate']
        self.min_count = data['min_count']
        self.max_count = data['max_count']

    def write(self, fname):
        """Write discount to a file"""
        try:
            import cPickle as pickle
        except:
            import pickle
        data = {'method': self.method,
                'discount': self.discount,
                'interpolate': self.interpolate,
                'min_count': self.min_count,
                'max_count': self.max_count}
        with open(fname, 'wb') as fout:
            pickle.dump(data, fout, 2)
