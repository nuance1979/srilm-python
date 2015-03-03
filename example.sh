#!/bin/bash

MT=$(../sbin/machine-type)
BIN=../bin/$MT

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 order vocab train heldout test [classes class_counts]"
    exit
fi

ORDER=$1
VOCAB=$2
TRAIN=$3
HELDOUT=$4
TEST=$5
CLASSES=$6
CLASSCOUNTS=$7

TMP=$(mktemp)
echo "Ngram LM with Good-Turing discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm - \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST
echo "Ngram LM with Witten-Bell discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm - -wbdiscount \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST
echo "Ngram LM with Kneser-Ney discount:"
for i in 1 2 3 4 5 6 7 8 9; do
    args="${args} -gt${i}min 1"
done
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm - -interpolate -ukndiscount $args \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST
echo "Ngram LM with Chen-Goodman discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm - -interpolate -kndiscount $args \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST
echo "Ngram LM with Jelinek-Mercer smoothing:"
COUNTS=$(mktemp)
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -write $COUNTS
$BIN/ngram-count -vocab $VOCAB -unk -write-vocab $TMP
VSIZE=$(wc $TMP | awk '{print $1}')
TCOUNT=$(perl -ane'$i+=$F[1] if (@F==2); END{print $i}' $COUNTS)
cat <<EOF > $TMP
order $ORDER
mixweights 3
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
countmodulus 1
vocabsize $VSIZE
totalcount $TCOUNT
counts $COUNTS
EOF
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $HELDOUT -init-lm $TMP1 -count-lm -lm - \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST -count-lm
echo "MaxEnt LM:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -maxent -lm - \
    | $BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm - -ppl $TEST -maxent
echo "Simple bi-gram class LM:"
$BIN/ngram-count -order 2 -unk -read $CLASSCOUNTS -lm - $args \
    | $BIN/ngram -order 2 -vocab $VOCAB -unk -lm - -ppl $TEST -classes $CLASSES -simple-classes
rm -f $TMP $COUNTS
