#!/usr/bin/env bash

MT="$(../sbin/machine-type)"
BIN="../bin/${MT}"

if [[ "$#" -lt 5 ]]; then
    echo "Usage: $(basename "$0") order vocab train heldout test"
    exit 1
fi

ORDER="$1"
VOCAB="$2"
TRAIN="$3"
HELDOUT="$4"
TEST="$5"

echo "Ngram LM with Good-Turing discount:"
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -lm - \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}"
echo "Ngram LM with Witten-Bell discount:"
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -lm - -wbdiscount \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}"
echo "Ngram LM with Kneser-Ney discount:"
args=""
for i in {1..9}; do
    args+=" -gt${i}min 1"
done
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -lm - -interpolate -ukndiscount ${args} \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}"
echo "Ngram LM with Chen-Goodman discount:"
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -lm - -interpolate -kndiscount ${args} \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}"
echo "Ngram LM with Jelinek-Mercer smoothing:"
TMP="$(mktemp)"
COUNTS="$(mktemp)"
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -write "${COUNTS}"
"${BIN}/ngram-count" -vocab "${VOCAB}" -unk -write-vocab "${TMP}"
wcarray=( $(wc "${TMP}") )
VSIZE="${wcarray[0]}"
TCOUNT="$(awk 'BEGIN { i=0 } NF==2 { i+=$2 } END { print i }' "${COUNTS}")"
cat <<EOF > "${TMP}"
order ${ORDER}
mixweights 3
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
 0.5 0.33 0.25
countmodulus 1
vocabsize ${VSIZE}
totalcount ${TCOUNT}
counts ${COUNTS}
EOF
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${HELDOUT}" -init-lm "${TMP}" -count-lm -lm - \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}" -count-lm
echo "MaxEnt LM:"
"${BIN}/ngram-count" -order "${ORDER}" -vocab "${VOCAB}" -unk -text "${TRAIN}" -maxent -lm - \
    | "${BIN}/ngram" -order "${ORDER}" -vocab "${VOCAB}" -unk -lm - -ppl "${TEST}" -maxent
rm -f "${TMP}" "${COUNTS}"
