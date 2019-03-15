#!/bin/sh
PROJECT="bert_extract"
#cd `dirname $0`
inputdate=$(date -d "$1" +%Y-%m-%d)
echo "execute $BASH_SOURCE for inputdate[$inputdate]"

model_dir="hdfs://xxxx:9000/home/hadoop/bert_extract/chinese_L-12_H-768_A-12.tgz"
BERT_BASE_DIR="./chinese_L-12_H-768_A-12"

INPUT="home/hadoop/bert_extract/data"
OUTPUT="home/hadoop/bert_extract/result"

pyfile=bert_extract.py
pyfile_path=./${pyfile}

PYPATH=hdfs://xxxx:9000/home/hadoop/python/anaconda2_tf0.12.zip
PYBIN=./anaconda2/bin/python2

SPARK_SUBMIT=/usr/bin/spark/bin/spark-submit 
${SPARK_SUBMIT} \
    --master yarn-cluster \
    --archives "${PYPATH}#anaconda2,${model_dir}#chinese_L-12_H-768_A-12" \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYBIN} \
    --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=${PYBIN} \
    --name  "Spark:[feature][bert_extract][${inputdate}]" \
    --conf spark.default.parallelism=6000 \
    --driver-memory 4G \
    --driver-cores 1 \
    --executor-memory 8G \
    --executor-cores 2 \
    --num-executors 20 \
    --queue root.default \
    --files "__init__.py,modeling.py,tokenization.py" \
    ${pyfile_path}  --input_dir=${INPUT} --output_dir=${OUTPUT}  --inputdate=${inputdate} --vocab_file=$BERT_BASE_DIR/vocab.txt --bert_config_file=$BERT_BASE_DIR/bert_config.json  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt --max_seq_length=38 --batch_size=8  --layers=-1 


