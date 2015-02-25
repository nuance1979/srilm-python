#/usr/bin/python

import os
import tempfile
import argparse
import srilm

def ngramLmWithGoodTuring(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'good-turing'))
    lm.train(tr)
    return lm.test(test)

def ngramLmWithWittenBell(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'witten-bell'))
    lm.train(tr)
    return lm.test(test)

def ngramLmWithKneserNey(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'kneser-ney', interpolate = True))
    lm.train(tr)
    return lm.test(test)

def ngramLmWithChenGoodman(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(i+1, srilm.discount.Discount(method = 'chen-goodman', interpolate = True))
    lm.train(tr)
    return lm.test(test)

def ngramSimpleClassLm(order, vocab, train, heldout, test, classes = None, class_count = None):
    lm = srilm.ngram.SimpleClassLm(vocab, 2)
    if not classes or not class_count:
        print "No class definition or class counts specified; inducing classes (this may take a while)...",
        tr = srilm.stats.Stats(vocab, 2)
        tr.count_file(train)
        fd, fname = tempfile.mkstemp()
        os.close(fd)
        fd, fcname = tempfile.mkstemp()
        os.close(fd)
        srilm.utils.train_class(tr, 1000, fname, fcname)
        print "done"
        lm.train(fname, fcname)
        prob, denom, ppl = lm.test(test)
        os.remove(fname)
        os.remove(fcname)
    else:
        lm.train(classes, class_count)
        prob, denom, ppl = lm.test(test)
    return (prob, denom, ppl)

def ngramCountLm(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.CountLm(vocab, order)
    lm.train(tr, heldout)
    return lm.test(test)

def maxentLm(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.maxent.Lm(vocab, order)
    lm.train(tr)
    return lm.test(test)

def main(args):
    vocab = srilm.vocab.Vocab()
    vocab.read(args.vocab)
    heldout = srilm.stats.Stats(vocab, args.order)
    heldout.count_file(args.heldout)
    test = srilm.stats.Stats(vocab, args.order)
    test.count_file(args.test)
    test.make_test()
    # we don't make a shared train stats because some model will change train stats during model estimation
    prob, denom, ppl = ngramLmWithGoodTuring(args.order, vocab, args.train, heldout, test)
    print 'Ngram LM with Good-Turing discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithWittenBell(args.order, vocab, args.train, heldout, test)
    print 'Ngram LM with Witten-Bell discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithKneserNey(args.order, vocab, args.train, heldout, test)
    print 'Ngram LM with Kneser-Ney discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramLmWithChenGoodman(args.order, vocab, args.train, heldout, test)
    print 'Ngram LM with Chen-Goodman discount: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramCountLm(args.order, vocab, args.train, heldout, test)
    print 'Ngram LM with Jelinek-Mercer smoothing: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = maxentLm(args.order, vocab, args.train, heldout, test)
    print 'MaxEnt LM: logprob =', prob, 'denom =', denom, 'ppl =', ppl
    prob, denom, ppl = ngramSimpleClassLm(args.order, vocab, args.train, heldout, test, args.classes, args.class_counts)
    print 'Simple bi-gram class LM: logprob =', prob, 'denom =', denom, 'ppl =', ppl

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
    parser.add_argument('--classes',
                        help = 'Class definition file')
    parser.add_argument('--class-counts', dest='class_counts',
                        help = 'Bigram class count file')
    args = parser.parse_args()
    main(args)
