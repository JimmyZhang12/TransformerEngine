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

#include <vector>

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

  int dim, myrank, nranks, _math_sms;
  ncclComm_t comm;
  std::vector<at::cuda::CUDAStream> _stream_compute;

  //TODO: CUDAStream doesnt have a default constuctor but instantiating here causes errors
  at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true); 

  cudaEvent_t _start_compute, _stop_compute, _stop_comm;


  NcclCommOverlap(int m, bool userbuffers);
  void RingExchange(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
               transformer_engine::DType A_type, bool transa, at::Tensor B,
               at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type,
               bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
               at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
               at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
               bool accumulate, bool use_split_accumulator, int comm_type, at::Tensor rs_output,
               bool debug_print);
  void print_tensor(torch::Tensor tensor);

  ~NcclCommOverlap();
};

}