#!/usr/bin/env bash

exp_path='/extra/yadongl10/BIG_sandbox/SparseImageFlows_result/lagan_pixelcnn/a1_b5'
log_path=${exp_path}/log

mkdir -p ${exp_path}
mkdir -p ${log_path}

qsub -q arcus.q -P arcus_gpu.p -l hostname=arcus-5 -l gpu=4 -o ${log_path}/exp0.out -e ${log_path}/exp0.err bash.sh

