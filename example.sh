#!/bin/bash

MT=$(../sbin/machine-type)
BIN=../bin/$MT

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 order vocab train heldout test"
    exit
fi

ORDER=$1
VOCAB=$2
TRAIN=$3
HELDOUT=$4
TEST=$5

TMP=$(mktemp)
echo "Ngram LM with Good-Turing discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm $TMP
$BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm $TMP -ppl $TEST
echo "Ngram LM with Witten-Bell discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm $TMP -wbdiscount
$BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm $TMP -ppl $TEST
echo "Ngram LM with Kneser-Ney discount:"
for i in 1 2 3 4 5 6 7 8 9; do
    args="${args} -gt${i}min 1"
done
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm $TMP -interpolate -ukndiscount $args
$BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm $TMP -ppl $TEST
echo "Ngram LM with Chen-Goodman discount:"
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -lm $TMP -interpolate -kndiscount $args
$BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm $TMP -ppl $TEST
echo "Ngram LM with Jelinek-Mercer smoothing:"
COUNTS=$(mktemp)
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $TRAIN -write $COUNTS
$BIN/ngram-count -vocab $VOCAB -unk -write-vocab $TMP
VSIZE=$(wc $TMP | awk '{print $1}')
TCOUNT=$(perl -ane'$i+=$F[@F-1] if (@F=='$ORDER'+1); END{print $i}' $COUNTS)
TMP1=$(mktemp)
echo $TMP1
cat <<EOF > $TMP1
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
echo $TMP
$BIN/ngram-count -order $ORDER -vocab $VOCAB -unk -text $HELDOUT -init-lm $TMP1 -count-lm -lm $TMP
$BIN/ngram -order $ORDER -vocab $VOCAB -unk -lm $TMP -ppl $TEST -count-lm
rm -f $TMP
