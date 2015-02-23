from cython.operator cimport dereference as deref
from stats cimport Stats

def rand_seed(long seed):
    srand48(seed)

def logprob_to_prob(LogP2 logprob):
    return LogPtoProb(logprob)

def prob_to_logprob(Prob prob):
    return ProbToLogP(prob)

def add_logprob(LogP2 x, LogP2 y):
    return AddLogP(x, y)

def train_class(Stats ts, unsigned num_class, const char *out_classes_fname, method = 'inc', exclude_list = ['<s>', '</s>']):
    if method not in ['full', 'inc']:
        raise ValueError('Invalid classing method; expect "full" or "inc"')
    if ts.order < 2:
        raise AttributeError('Invalid order for stats; expect >= 2')
    cdef SubVocab *class_vocab_ptr = new SubVocab(deref(ts._vocab.thisptr), False)
    cdef UniqueWordClasses *classing = new UniqueWordClasses(deref(ts._vocab.thisptr), deref(class_vocab_ptr))
    cdef SubVocab *exclude_vocab_ptr = new SubVocab(deref(ts._vocab.thisptr), False)
    for w in exclude_list:
        exclude_vocab_ptr.addWord(ts._vocab.thisptr.getIndex(w))
    classing.initialize(deref(ts.thisptr), deref(exclude_vocab_ptr))
    if method == 'full':
        classing.fullMerge(num_class)
    else:
        classing.incrementalMerge(num_class)
    cdef File *fptr = new File(out_classes_fname, 'w', 0)
    if fptr == NULL:
        raise MemoryError
    classing.writeClasses(deref(fptr))
    del fptr
    del exclude_vocab_ptr
    del classing
    del class_vocab_ptr
