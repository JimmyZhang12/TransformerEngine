#!/bin/bash

unset SLURM_NTASKS
SLURM_JOB_NAME=bash

# nvidia-smi | grep 'python' | awk '{ print $5 }'  | xargs -n1 kill -9

TE3='/lustre/fsw/portfolios/llmservice/users/jiemingz/ncclcomms/TransformerEngine/'
TE2='/lustre/fsw/portfolios/llmservice/users/jiemingz/ncclcomms/TransformerEngine/transformer_engine'
TE1='/lustre/fsw/portfolios/llmservice/users/jiemingz/ncclcomms/TransformerEngine/build/lib.linux-x86_64-cpython-310'
export PYTHONPATH=${MLM}:${TE3}:${TE2}:${TE}:${NEMO}:/opt/NeMo-Megatron-Launcher/launcher_scripts;
export NCCL_AVOID_RECORD_STREAMS=1
export UCX_TLS=self,tcp

NSYS="nsys profile -s none -o /lustre/fsw/portfolios/llmservice/users/jiemingz/ncclcomms/profiles/profile_yesoverlap.%q{OMPI_COMM_WORLD_RANK}  -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop "
# export NCCL_NVLS_ENABLE=1
# export NCCL_DEBUG_SUBSYS=NVLS
# export NCCL_DEBUG=INFO 


# NCCL_MAX_NCHANNELS
# export NVTE_EXT_MARGIN_SM=${COMM_SMS}
# export NCCL_MAX_CTAS=${COMM_SMS}
# export NCCL_MIX_CTAS=${COMM_SMS}

export CUBLASLT_LOG_LEVEL=1

COMM_SMS=16
NCHANNELS=16

NCCL_MIN_NCHANNELS=${NCHANNELS} \
NCCL_MAX_NCHANNELS=${NCHANNELS} \
NVTE_EXT_MARGIN_SM=${COMM_SMS} \
NCCL_MAX_CTAS=${COMM_SMS} \
NCCL_MIX_CTAS=${COMM_SMS} \
mpirun -n 4 --allow-run-as-root --oversubscribe ${NSYS} python -u /lustre/fsw/portfolios/llmservice/users/jiemingz/ncclcomms/nccl_tests.py
