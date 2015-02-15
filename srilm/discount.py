import cPickle as pickle

class Discount(object):
    """Hold parameters for Ngram discount/smoothing method

    Note that you need one Discount() obj for each ngram order. If Discount.discount 
    is provided, it will be used in Lm.train(); otherwise, it will be set with the
    estimated value.
    """
    def __init__(self, method = None, discount = None, interpolate=None, min_count = None, max_count = None):
        if method is not None and method not in ['kneser-ney', 'good-turing', 'witten-bell', 'chen-goodman']:
            raise ValueError('Unknown smoothing method: '+ method)
        self.method = method
        if discount is not None and not isinstance(discount, (int, long, float)):
            raise ValueError('Expect a number for discount')
        self.discount = discount
        if interpolate is not None and not isinstance(interpolate, bool):
            raise ValueError('Expect a bool for interpolate')
        self.interpolate = interpolate
        if min_count is not None and not isinstance(min_count, (int, long)):
            raise ValueError('Expect a number for min_count')
        self.min_count = min_count
        if max_count is not None and not isinstance(max_count, (int, long)):
            raise ValueError('Expect a number for max_count')
        self.max_count = max_count

    def read(self, fname):
        with open(fname, 'rb') as f:
            temp_dict = pickle.load(f)
        self.__dict__.update(temp_dict)

    def write(self, fname):
        with open(fname, 'wb') as fout:
            pickle.dump(self.__dict__, fout, 2)
