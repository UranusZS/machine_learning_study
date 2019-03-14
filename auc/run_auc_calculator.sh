#!/bin/sh
PROJECT="auc_calculator"
#cd `dirname $0`
inputdate=$(date -d "$1" +%Y-%m-%d)
echo "execute $BASH_SOURCE for inputdate[$inputdate]"

INPUT="/home/hadoop/data/predict"
OUTPUT="/home/hadoop/data/auc"

pyfile=auc_calculator.py
pyfile_path=./python/${pyfile}

SPARK_SUBMIT=/usr/bin//spark/bin/spark-submit 
${SPARK_SUBMIT} \
    --master yarn-cluster \
    --name  "Spark:[evaluate][auc_metric][${inputdate}]" \
    --conf spark.default.parallelism=6000 \
    --driver-memory 4G \
    --driver-cores 1 \
    --executor-memory 8G \
    --executor-cores 2 \
    --num-executors 2 \
    --queue root.default \
    ${pyfile_path}  --input_dir=${INPUT} --output_dir=${OUTPUT}  --inputdate=${inputdate} 
