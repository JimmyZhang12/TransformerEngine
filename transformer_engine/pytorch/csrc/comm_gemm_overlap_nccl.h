/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <torch/cuda.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/types.h>

#include "common/util/logging.h"
#include "common/util/system.h"

#include <nccl.h>
#include <mpi.h>


#define HALF_BYTES 2
#define UB_MAX_SM 32

#define CHECK_CUDA(call)                                                                           \
  do {                                                                                             \
    cudaError_t status_ = call;                                                                    \
    if (status_ != cudaSuccess) {                                                                  \
      fprintf(stderr, "CUDA Error at line %d: %s\n", __LINE__, cudaGetErrorString(status_));       \
      exit(1);                                                                                     \
    }                                                                                              \
  } while (0)

#define NCCLCHECK(cmd)                                                                                                 \
  do                                                                                                                 \
  {                                                                                                                  \
      ncclResult_t r = cmd;                                                                                          \
      if (r != ncclSuccess)                                                                                          \
      {                                                                                                              \
          printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
          exit(1);                                                                                        \
      }                                                                                                              \
  } while (0)


using namespace torch::indexing;
namespace nccl_ubuf {

struct NcclCommOverlap : torch::CustomClassHolder{
  void* sendbuf;
  void* recvbuf;
  void* sendRegHandle;
  void* recvRegHandle;
  bool _userbuffers;

  int dim, myrank, nranks;
  ncclComm_t comm;

  NcclCommOverlap(int m, bool userbuffers);
  void RingExchange(bool debug_print);
  void print_tensor(torch::Tensor tensor);

  ~NcclCommOverlap();
};

}