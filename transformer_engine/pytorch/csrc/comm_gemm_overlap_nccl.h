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

enum class COMM_TYPE { RS = 0, AG = 1 };

enum class UBOverlapAlgo {
  BULK_OVERLAP_AG = 0,
  BULK_OVERLAP_RS = 1,
  SPLIT_PIPELINED_AG_P2P = 2,
  SPLIT_PIPELINED_RS = 3,
  SPLIT_PIPELINED_RS_P2P = 4,
  ATOMIC_GEMM_RS = 5,
  ATOMIC_GEMM_AG_P2P = 6,
  ATOMIC_GEMM_RS_P2P = 7
};

struct NcclCommOverlap : torch::CustomClassHolder{
  bool _userbuffers;
  bool _overlap;

  void* _ubuf_ptr;
  void* _ubuf_handle;
  at::Tensor _ubuf;
  std::vector<torch::Tensor> _ubufs;

  torch::Tensor _ubuf_scale_inv;
  bool _ubuf_scale_inv_initialized;
  bool _atomic_gemm = false;
  int _self_chunk_id, _rank_round_tp, _num_max_streams;

  int _num_chunks, _tp_id, _tp_size, _total_sms, _math_sms, _next_rank, _prev_rank, rank;
  ncclComm_t comm;
  std::vector<at::cuda::CUDAStream> _stream_compute;

  //TODO: CUDAStream doesnt have a default constuctor but instantiating here causes errors
  at::cuda::CUDAStream _stream_comm = at::cuda::getStreamFromPool(true); 
  cudaEvent_t _start_compute, _stop_compute, _stop_comm, _start_comm;

  NcclCommOverlap(torch::Tensor sample, 
                int rank, int tp_size, int num_comm_sm, int num_max_streams, 
                bool set_sm_margin, bool aggregate2, bool is_reduce_scatter, 
                bool atomic_gemm, bool userbuffers);


  torch::Tensor split_overlap_ag_p2p(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
               transformer_engine::DType A_type, bool transa, at::Tensor B,
               at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type,
               bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
               at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
               at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
               bool accumulate, bool use_split_accumulator, at::Tensor B_copy);

  void split_overlap_rs_p2p(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type, 
                        bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, 
                        at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, 
                        bool accumulate, bool use_split_accumulator, at::Tensor rs_output);

  std::vector<at::Tensor> bulk_overlap(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
               transformer_engine::DType A_type, bool transa, at::Tensor B,
               at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type,
               bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
               at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
               at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
               bool accumulate, bool use_split_accumulator, int comm_type, at::Tensor rs_output); 

  void playground(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                        transformer_engine::DType A_type, bool transa, at::Tensor B,
                        at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type, 
                        bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type, 
                        at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
                        at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize, 
                        bool accumulate, bool use_split_accumulator, at::Tensor rs_output);

  void copy_input_to_ubuf(torch::Tensor input, bool chunk);
  torch::Tensor get_ubuf_output(int comm_type);

  template <typename fp8type>
  void reduce_fp8_in_bf16_out(void *inputs, void *output, float *scale, int num_inputs,
                              int input_size, cudaStream_t stream);
  bool is_fp8_ubuf() { return (_ubuf.element_size() == 1); }
  bool is_atomic_gemm() { return _atomic_gemm; }
  bool is_p2p_overlap() { return true; }
  void set_ubuf_scale_inv(const torch::Tensor &scale_inv) {
    _ubuf_scale_inv = scale_inv;
    _ubuf_scale_inv_initialized = true;
  }

  ~NcclCommOverlap();
};

}