#!/usr/bin/env python3

import argparse
import srilm.vocab
import srilm.stats
import srilm.ngram
import srilm.discount
import srilm.maxent

# magic min/max numbers; see ngram-count.cc
gtmin = [1, 1, 1, 2, 2, 2, 2, 2, 2, 2]
gtmax = [5, 1, 7, 7, 7, 7, 7, 7, 7, 7]


def ngramLmWithGoodTuring(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(
            i + 1,
            srilm.discount.Discount(
                method="good-turing", min_count=gtmin[i + 1], max_count=gtmax[i + 1]
            ),
        )
    lm.train(tr)
    return lm.test(test)


def ngramLmWithWittenBell(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(
            i + 1, srilm.discount.Discount(method="witten-bell", min_count=gtmin[i + 1])
        )
    lm.train(tr)
    return lm.test(test)


def ngramLmWithKneserNey(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(
            i + 1, srilm.discount.Discount(method="kneser-ney", interpolate=True)
        )
    lm.train(tr)
    return lm.test(test)


def ngramLmWithChenGoodman(order, vocab, train, heldout, test):
    tr = srilm.stats.Stats(vocab, order)
    tr.count_file(train)
    lm = srilm.ngram.Lm(vocab, order)
    for i in range(order):
        lm.set_discount(
            i + 1, srilm.discount.Discount(method="chen-goodman", interpolate=True)
        )
    lm.train(tr)
    return lm.test(test)


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
    prob, denom, ppl = ngramLmWithGoodTuring(
        args.order, vocab, args.train, heldout, test
    )
    print(
        "Ngram LM with Good-Turing discount: logprob =",
        prob,
        "denom =",
        denom,
        "ppl =",
        ppl,
    )
    prob, denom, ppl = ngramLmWithWittenBell(
        args.order, vocab, args.train, heldout, test
    )
    print(
        "Ngram LM with Witten-Bell discount: logprob =",
        prob,
        "denom =",
        denom,
        "ppl =",
        ppl,
    )
    prob, denom, ppl = ngramLmWithKneserNey(
        args.order, vocab, args.train, heldout, test
    )
    print(
        "Ngram LM with Kneser-Ney discount: logprob =",
        prob,
        "denom =",
        denom,
        "ppl =",
        ppl,
    )
    prob, denom, ppl = ngramLmWithChenGoodman(
        args.order, vocab, args.train, heldout, test
    )
    print(
        "Ngram LM with Chen-Goodman discount: logprob =",
        prob,
        "denom =",
        denom,
        "ppl =",
        ppl,
    )
    prob, denom, ppl = ngramCountLm(args.order, vocab, args.train, heldout, test)
    print(
        "Ngram LM with Jelinek-Mercer smoothing: logprob =",
        prob,
        "denom =",
        denom,
        "ppl =",
        ppl,
    )
    prob, denom, ppl = maxentLm(args.order, vocab, args.train, heldout, test)
    print("MaxEnt LM: logprob =", prob, "denom =", denom, "ppl =", ppl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train various types of language models on the same train/heldout/test data"
    )
    parser.add_argument("--order", type=int, default=3, help="Order of the model")
    parser.add_argument("--vocab", required=True, help="Vocabulary file")
    parser.add_argument("--train", required=True, help="Training text file")
    parser.add_argument("--heldout", required=True, help="Heldout text file")
    parser.add_argument("--test", required=True, help="Test text file")
    args = parser.parse_args()
    main(args)
