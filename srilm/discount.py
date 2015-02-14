class Discount(object):
    """Hold parameters for Ngram discount/smoothing method"""
    def __init__(self, method, discounts=None, min_counts=None, max_counts=None):
        if method not in ['kneser-ney', 'good-turing', 'witten-bell', 'chen-goodman']:
            raise ValueError('Unknown smoothing method: '+ method)
        self.method = method
        self.discounts = []
        if discounts is not None:
            try:
                for i in range(len(discounts)):
                    self.discounts[i] = discounts[i]
            except:
                raise ValueError('Invalid value for discounts; expect list of numbers')
        self.min_counts = []
        if min_counts is not None:
            try:
                for i in range(len(min_counts)):
                    assert isinstance(min_counts[i], (int, long, float))
                    self.min_counts.append(min_counts[i])
            except:
                raise ValueError('Invalid value for min_counts; expect list of numbers')
        self.max_counts = []
        if max_counts is not None:
            try:
                for i in range(len(max_counts)):
                    assert isinstance(max_counts[i], (int, long, float))
                    self.max_counts.append(max_counts[i])
            except:
                raise ValueError('Invalid value for max_counts; expect list of numbers')
