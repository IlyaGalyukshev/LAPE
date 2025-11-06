#!/bin/bash
# Script to compare two model analysis results

# Define parameters
LANG1="crh_Latn"
LANG2="eng"
MODEL1="tweety"
MODEL2="Mistral"
MODEL1_NAME="Tweety-Tatar"
MODEL2_NAME="Mistral"

# Construct file paths
JSON1="../activation_results/${LANG1}_${LANG2}_${MODEL1}.json"
JSON2="../activation_results/${LANG1}_${LANG2}_${MODEL2}.json"
OUTPUT="../comparison/${MODEL1}_vs_${MODEL2}_${LANG1}_${LANG2}"


python3 compare_models.py \
    --json1 ${JSON1} \
    --json2 ${JSON2} \
    --name1 "${MODEL1_NAME}" \
    --name2 "${MODEL2_NAME}" \
    --output ${OUTPUT}

