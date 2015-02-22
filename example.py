#/usr/bin/python

import argparse
import srilm

def ngramLmWithGoodTuring(order, vocab, train, heldout, test):
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'good-turing'))
    lm.train(train)
    return lm.test(test)

def ngramLmWithWittenBell(order, vocab, train, heldout, test):
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'witten-bell'))
    lm.train(train)
    return lm.test(test)

def ngramLmWithKneserNey(order, vocab, train, heldout, test):
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'kneser-ney', interpolate = True))
    lm.train(train)
    return lm.test(test)

def ngramLmWithChenGoodman(order, vocab, train, heldout, test):
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'chen-goodman', interpolate = True))
    lm.train(train)
    return lm.test(test)

def ngramClassLm(order, vocab, train, heldout, test):
    lm = srilm.ngram.ClassLm(vocab, order)
    lm.train_class(train, num_class = 100)
    lm.train(train)
    return lm.test(test)

def maxentLm(order, vocab, train, heldout, test):
    lm = srilm.maxent.Lm(vocab, order)
    lm.train(train)
    return lm.test(test)

def main(args):
    vocab = srilm.vocab.Vocab()
    vocab.read(args.vocab)
    train = srilm.stats.Stats(vocab, args.order)
    train.count_file(args.train)
    heldout = srilm.stats.Stats(vocab, args.order)
    heldout.count_file(args.heldout)
    test = srilm.stats.Stats(vocab, args.order)
    test.count_file(args.test)
    test.make_test()
    prob, denom, ppl = ngramLmWithGoodTuring(args.order, vocab, train, heldout, test)
    print 'Ngram LM with Good-Turing discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithWittenBell(args.order, vocab, train, heldout, test)
    print 'Ngram LM with Witten-Bell discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithKneserNey(args.order, vocab, train, heldout, test)
    print 'Ngram LM with Kneser-Ney discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithChenGoodman(args.order, vocab, train, heldout, test)
    print 'Ngram LM with Chen-Goodman discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = maxentLm(args.order, vocab, train, heldout, test)
    print 'MaxEnt LM: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = maxentLm(args.order, vocab, train, heldout, test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train various types of language models on the same train/heldout/test data')
    parser.add_argument('--order', type = int, default = 3,
                        help = 'Order of the model')
    parser.add_argument('--vocab', required = True,
                        help = 'Vocabulary file')
    parser.add_argument('--train', required = True,
                        help = 'Training text file')
    parser.add_argument('--heldout', required = True,
                        help = 'Heldout text file')
    parser.add_argument('--test', required = True,
                        help = 'Test text file')
    args = parser.parse_args()
    main(args)
