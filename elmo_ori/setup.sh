#!/bin/bash

# python2 package
# PYTHON_FILE=/app/idl/users/dl/liuyuan/zhangbiao04/python.tar.gz

# marco data file from Zhang Biao
# DATA_FILE=/app/idl/users/dl/liuyuan/zhangbiao04/marco.v1.0.1.tar.gz

# python3 package from Sun Xingwu
PYTHON_PKG="python3.5.1.tar.gz"
PYTHON_FILE="/app/idl/users/dl/liuyuan/lihongyu04/tools/$PYTHON_PKG"

# Baike data file
DATA_PKG="baike_mini.tar"
DATA_FILE="/app/idl/users/dl/liuyuan/lihongyu04/baike_data/$DATA_PKG"


hadoop fs \
    -D fs.default.name=hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 \
    -D mapred.job.tracker=nmg01-mulan-job.dmop.baidu.com:54311 \
    -D hadoop.job.ugi=idl-dl,idl-dl@admin \
    -get $DATA_FILE .

hadoop fs \
    -D fs.default.name=hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 \
    -D mapred.job.tracker=nmg01-mulan-job.dmop.baidu.com:54311 \
    -D hadoop.job.ugi=idl-dl,idl-dl@admin \
    -get $PYTHON_FILE .

tar -xzvf $PYTHON_PKG
tar -xvf $DATA_PKG
