#!/bin/bash

if [ "$#" -lt 1 ]; then
    echo "Wrong submission format"
    exit 1
fi

echo 'Start task: '$1

TASK_DIR=ELMo
JOB_NAME=$1
HADOOP_WORK_DIR=/app/idl/users/dl/liuyuan/lihongyu04/$TASK_DIR/$JOB_NAME
QUEUE_NAME=yq01-p40-3-8
HGCP_CLIENR_BIN=~/.hgcp/software-install/HGCP_client/bin
/home/ssd4/dqa/wangyizhong/hadoop-v2/hadoop/bin/hadoop fs -rmr $HADOOP_WORK_DIR

${HGCP_CLIENR_BIN}/submit \
    --hdfs hdfs://nmg01-mulan-hdfs.dmop.baidu.com:54310 \
    --hdfs-user idl-dl \
    --hdfs-passwd idl-dl@admin \
    --hdfs-path $HADOOP_WORK_DIR \
    --file-dir ./ \
    --job-name $JOB_NAME \
    --queue-name $QUEUE_NAME \
    --num-nodes 1 \
    --num-task-pernode 1 \
    --gpu-pnode 8 \
    --time-limit 100000 \
    --submitter lihongyu04 \
    --job-script ./job.sh
