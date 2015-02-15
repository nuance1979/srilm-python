import cPickle as pickle

class Discount(object):
    """Hold parameters for Ngram discount/smoothing method

    Note that you need one Discount() obj for each ngram order. If Discount.discount 
    is provided, it will be used in Lm.train(); otherwise, it will be set with the
    estimated value.
    """
    def __init__(self, method = None, discount = None, min_count = None, max_count = None):
        if method is not None and method not in ['kneser-ney', 'good-turing', 'witten-bell', 'chen-goodman']:
            raise ValueError('Unknown smoothing method: '+ method)
        self.method = method
        if discount is not None and not isinstance(discount, (int, long, float)):
            raise ValueError('Expect a number')
        if min_count is not None and not isinstance(min_count, (int, long, float)):
            raise ValueError('Expect a number')
        if max_count is not None and not isinstance(max_count, (int, long, float)):
            raise ValueError('Expect a number')

    def read(self, fname):
        with open(fname, 'r') as f:
            self = pickle.load(f)

    def write(self, fname):
        with open(fname, 'w') as fout:
            pickle.dump(self, fout)
