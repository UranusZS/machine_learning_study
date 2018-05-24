export PYTHONPATH=./:$PYTHONPATH
export LD_LIBRARY_PATH=./tensorflow/cuda/lib64:$LD_LIBRARY_PATH

#export GRPC_TRACE=all 
export GRPC_VERBOSITY=DEBUG
export HADOOP_HDFS_HOME=/usr/bin/hadoop/software/hadoop/
#source ${HADOOP_HOME}/libexec/hadoop-config.sh
#CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob) python text_classification_cnn.py 

python text_classification_cnn.py --is_distribution=true --num_epochs=2 
