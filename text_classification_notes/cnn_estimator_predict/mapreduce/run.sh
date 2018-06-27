#!/bin/sh

_date="20180513"
if ! test -z "$1"
then
    _date=$1
else
    _date=`date -d-1day +\%Y\%m\%d`
fi

PYPATH=hdfs://xxxx:9000/xxx/python/anaconda2.zip
PYBIN=./anaconda2/bin/python2

INPUT=/xxx/preprocess/${_date}
OUTPUT=/xxx/mr_result/${_date}

MODEL_ZIP=hdfs://xxx/model/export_model_zip

HADOOP_DIR=/usr/bin/hadoop
HADOOP_BIN=${HADOOP_DIR}/bin/hadoop
STREAMING_JAR=${HADOOP_DIR}/contrib/streaming/hadoop-streaming.jar

#${HADOOP_BIN} jar ${STREAMING_JAR} \
${HADOOP_BIN} streaming \
    -D mapred.job.name="predict" \
    -D mapred.job.queue.name=root.default \
    -D mapred.map.tasks=50 \
    -D mapreduce.map.memory.mb=6000 \
    -D mapred.max.map.failures.percent=50 \
    -D mapred.min.split.size=1073741824 \
    -D mapreduce.task.timeout=9600000 \
    -D mapred.reduce.tasks=10 \
    -cacheArchive ${PYPATH}#anaconda2 \
    -cacheFile ${MODEL_ZIP}#export_model.zip \
    -input ${INPUT} \
    -output ${OUTPUT} \
    -file mapper.py \
    -file reducer.py \
    -file lib/utils.py \
    -file input.schema \
    -mapper "${PYBIN} mapper.py" \
    -reducer "${PYBIN} reducer.py" 

