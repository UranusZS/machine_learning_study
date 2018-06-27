#!/bin/sh

_date="20180513"
if ! test -z "$1"
then
    _date=$1
else
    _date=`date -d-1day +\%Y\%m\%d`
fi

SPARK_SUBMIT=/xxx/spark/bin/spark-submit
PYPATH=hdfs://xxx:9000/xxx/python/anaconda2.zip
PYBIN=./anaconda2/bin/python2

INPUT=/xxx/preprocess/${_date}
OUTPUT=/xxx/result/${_date}


${SPARK_SUBMIT} \
    --master yarn-cluster \
    --archives ${PYPATH}#anaconda2 \
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=${PYBIN} \
    --conf spark.yarn.appMasterEnv.PYSPARK_DRIVER_PYTHON=${PYBIN} \
    --driver-memory 4G \
    --driver-cores 1 \
    --executor-memory 5G \
    --executor-cores 1 \
    --num-executors 200 \
    --py-files lib.zip,export_model.zip \
    --queue root.default \
    --files input.schema \
    tf_predict.py --input_dir=${INPUT} --output_dir=${OUTPUT}
