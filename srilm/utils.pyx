"""
Module contains utility functions
"""

from cython.operator cimport dereference as deref
from stats cimport Stats

def rand_seed(long seed):
    """Set the seed of the random number generator"""
    srand48(seed)

def logprob_to_prob(LogP2 logprob):
    """Convert a base-10 log probability to a probability"""
    return LogPtoProb(logprob)

def prob_to_logprob(Prob prob):
    """Convert a probability to a base-10 log probability"""
    return ProbToLogP(prob)

def add_logprob(LogP2 x, LogP2 y):
    """Add two base-10 log probabilities"""
    return AddLogP(x, y)

def sub_logprob(LogP2 x, LogP2 y):
    """Substract base-10 log probability y from x""" 
    return SubLogP(x, y)

def mix_logprob(LogP2 prob1, LogP2 prob2, double lambda1):
    """Compute the interpolation of two base-10 log probabilites

    Return the equivalent of 
    prob_to_logprob(lambda1 * logprob_to_prob(prob1) + (1-lambda1) * logprob_to_prob(prob2))
    """
    return MixLogP(prob1, prob2, lambda1)

def train_class(Stats ts, unsigned num_class, out_classes, out_class_counts, method = 'inc', exclude_list = ['<s>', '</s>']):
    """Train an agglomerative clustering, aka, Brown clustering from a word bigram Stats

    Write out class definition and class bigram counts to files.
    """
    if method not in ['full', 'inc']:
        raise ValueError('Invalid classing method; expect "full" or "inc"')
    if ts.order < 2:
        raise AttributeError('Invalid order for stats; expect >= 2')
    cdef SubVocab *class_vocab_ptr = new SubVocab(deref(ts._vocab.thisptr), False)
    cdef UniqueWordClasses *classing = new UniqueWordClasses(deref(ts._vocab.thisptr), deref(class_vocab_ptr))
    cdef SubVocab *exclude_vocab_ptr = new SubVocab(deref(ts._vocab.thisptr), False)
    for w in exclude_list:
        exclude_vocab_ptr.addWord(ts._vocab.thisptr.getIndex(w.encode()))
    classing.initialize(deref(ts.thisptr), deref(exclude_vocab_ptr))
    if method == 'full':
        classing.fullMerge(num_class)
    else:
        classing.incrementalMerge(num_class)
    cdef File *fptr = new File(out_classes.encode(), b'w', 0)
    if fptr == NULL:
        raise MemoryError
    classing.writeClasses(deref(fptr))
    del fptr
    cdef File *fcptr = new File(out_class_counts.encode(), b'w', 0)
    if fcptr == NULL:
        raise MemoryError
    classing.writeCounts(deref(fcptr))
    del fcptr
    del exclude_vocab_ptr
    del classing
    del class_vocab_ptr
