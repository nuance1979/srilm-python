def rand_seed(long seed):
    srand48(seed)

def logprob_to_prob(LogP2 logprob):
    return LogPtoProb(logprob)

def prob_to_logprob(Prob prob):
    return ProbToLogP(prob)

def add_logprob(LogP2 x, LogP2 y):
    return AddLogP(x, y)
