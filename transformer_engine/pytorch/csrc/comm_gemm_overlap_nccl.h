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
#include "userbuffers/userbuffers.h"

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

using namespace torch::indexing;
namespace nccl_ubuf {

struct NcclCommOverlap : torch::CustomClassHolder{

  NcclCommOverlap(int num){
    MPI_Init(NULL, NULL);


    int myrank = -1;
    int nranks = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    printf("HelloWorld- myrank:%d ranks: %d\n", myrank, nranks);
    
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;

    if (myrank == 0){
      for (int i=1; i<nranks; i++){
        MPI_Send(&id, sizeof(id), MPI_BYTE, i, 0, MPI_COMM_WORLD);
      }
    }
    else{
      MPI_Status status;
      MPI_Recv(&id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
    }

    ncclCommInitRank(&comm, nranks, id, myrank);

    auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);

    if (myrank == 0){
      for (int i=1; i<nranks; i++){
        torch::Tensor rand_tensor = torch::randn({10}, options);
        ncclSend(rand_tensor.data_ptr<float>(), 10, ncclFloat, i, comm, cudaStream_t());
      }
    }
    else{
      ncclRecv(receiveTensor.data_ptr<float>(), 10, ncclFloat, 0, comm, cudaStream_t());

    }




    ncclCommDestroy(comm);
    MPI_Finalize();

  }
  torch::Tensor split_overlap_ag(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                                 transformer_engine::DType A_type, bool transa, at::Tensor B,
                                 at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                                 transformer_engine::DType B_type, bool transb, at::Tensor D,
                                 at::Tensor D_scale, transformer_engine::DType D_type,
                                 at::Tensor D_amax, at::Tensor bias,
                                 transformer_engine::DType bias_type, at::Tensor pre_gelu_out,
                                 bool grad, at::Tensor workspace, size_t workspaceSize,
                                 bool accumulate, bool use_split_accumulator, at::Tensor B_copy) {

    // Get GEMM dimensions between TN and NN input layouts
    const int m = (transa) ? A.size(0) : A.size(1);
    const int k = (transa) ? A.size(1) : A.size(0);
    // const int n_chunk = _ubufs[0].size(0);

    // Get communication and GEMM output chunk sizes
    // const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
    // const int output_chunk_bytes = (do_gelu
    //                                 ? (n_chunk * m) * D.element_size()
    //                                 : (n_chunk * m) * HALF_BYTES);
    // const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out.element_size() : 0;

    // // Get output and workspace data pointers
    // char *output_ptr = reinterpret_cast<char *>(D.data_ptr());
    // char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out.data_ptr());
    // char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
    // int workspace_size_chunk = workspaceSize / _stream_compute.size();

    // at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    // CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
    // CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _start_compute, 0));
    // CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_recv, _start_compute, 0));
    // for (int i = 0; i < _stream_compute.size(); i++) {
    //   CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i], _start_compute, 0));
    // }
    // if (_aggregate2) {
    //   const int num_steps = _tp_size / 2;
    //   char *input_b_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());

    //   // Initial 1X input chunk exchange between neighboring peers
    //   int send_chunk_id = _tp_id;
    //   int recv_chunk_id = (_tp_id % 2 == 0) ? _tp_id + 1 : _tp_id - 1;
    //   int send_offset = comm_bytes * send_chunk_id;
    //   int recv_offset = comm_bytes * recv_chunk_id;
    //   int peer_rank = (_tp_id % 2 == 0) ? _next_rank : _prev_rank;
    //   userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm, peer_rank,
    //                    (cudaStream_t)_stream_send);
    //   userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm, peer_rank,
    //                    (cudaStream_t)_stream_recv);
    //   CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    //   CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
    //   CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _stop_recv, 0));

    //   int local_rank_round2 = (_tp_id % 2 == 0) ? _tp_id : _tp_id - 1;
    //   const int next_rank = (_tp_size + _tp_id + 2) % _tp_size + _rank_round_tp;
    //   const int prev_rank = (_tp_size + _tp_id - 2) % _tp_size + _rank_round_tp;

    //   // Ring exchange of 2X inputs chunks
    //   for (int i = 0; i < num_steps; i++) {
    //     send_chunk_id = (_tp_size + local_rank_round2 - i * 2) % _tp_size;
    //     recv_chunk_id = (_tp_size + local_rank_round2 - i * 2 - 2) % _tp_size;
    //     send_offset = comm_bytes * send_chunk_id;
    //     recv_offset = comm_bytes * recv_chunk_id;

    //     // GEMM
    //     torch::Tensor input_b_chunk =
    //         torch::from_blob(input_b_ptr + send_offset, {n_chunk * 2, k}, _ubuf.options());
    //     torch::Tensor output_chunk = torch::from_blob(
    //         output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk * 2, m}, D.options());
    //     if (do_gelu) {
    //       pre_gelu_out = torch::from_blob(
    //           pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes),
    //           {n_chunk * 2, m},
    //           pre_gelu_out.options());
    //     }
    //     torch::Tensor workspace_chunk =
    //         torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
    //                          {workspace_size_chunk}, workspace.options());
    //     at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
    //     te_gemm(A, A_scale_inverse, A_type, transa, input_b_chunk, B_scale_inverse, B_type, transb,
    //             output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
    //             workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
    //             _math_sms);

    //     if (i < num_steps - 1) {
    //       // P2P communication
    //       userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes * 2, _ub_comm,
    //                        next_rank, (cudaStream_t)_stream_send);
    //       userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes * 2, _ub_comm,
    //                        prev_rank, (cudaStream_t)_stream_recv);
    //       CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    //       CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
    //       CHECK_CUDA(cudaStreamWaitEvent(
    //           (cudaStream_t)_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
    //     } else if (B_copy.numel() > 0) {
    //       assert(B_copy.numel() == _ubufs[_tp_id].numel());
    //       assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
    //       CHECK_CUDA(cudaMemcpyAsync(B_copy.data_ptr(), _ubufs[_tp_id].data_ptr(),
    //                                  _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
    //                                  cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_send));
    //     }
    //   }
    // } else {
    // for (int i = 0; i < _tp_size; i++) {
    //   // Set the userbuffer id. Buffer under send is the input for the current
    //   // GEMM chunk The initial input chunk is stored _ubuf[rank]. This is to
    //   // have the AG output in all ranks to be contiguous after the ring
    //   // exchanges
    //   int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
    //   int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
    //   int send_offset = comm_bytes * send_chunk_id;
    //   int recv_offset = comm_bytes * recv_chunk_id;

    //   // GEMM
    //   torch::Tensor output_chunk = torch::from_blob(
    //       output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk, m}, D.options());

    //   torch::Tensor workspace_chunk =
    //       torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
    //                         {workspace_size_chunk}, workspace.options());
    //   at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
    //   te_gemm(A, A_scale_inverse, A_type, transa, _ubufs[send_chunk_id], B_scale_inverse, B_type,
    //           transb, output_chunk, D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
    //           workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
    //           _math_sms);

    //   if (i < _tp_size - 1) {
    //     // P2P communication
    //     userbuffers_send(_ub_reg, send_offset, _ub_reg, send_offset, comm_bytes, _ub_comm,
    //                       _next_rank, (cudaStream_t)_stream_send);
    //     userbuffers_recv(_ub_reg, recv_offset, _ub_reg, recv_offset, comm_bytes, _ub_comm,
    //                       _prev_rank, (cudaStream_t)_stream_recv);
    //     CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    //     CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_send, _stop_recv, 0));
    //     CHECK_CUDA(cudaStreamWaitEvent(
    //         (cudaStream_t)_stream_compute[(i + 1) % _stream_compute.size()], _stop_recv, 0));
    //   } else if (B_copy.numel() > 0) {
    //     assert(B_copy.numel() == _ubufs[_tp_id].numel());
    //     assert(B_copy.element_size() == _ubufs[_tp_id].element_size());
    //     CHECK_CUDA(cudaMemcpyAsync(B_copy.data_ptr(), _ubufs[_tp_id].data_ptr(),
    //                                 _ubufs[_tp_id].numel() * _ubufs[_tp_id].element_size(),
    //                                 cudaMemcpyDeviceToDevice, (cudaStream_t)_stream_send));
    //   }
    // }
  
    // for (int i = 0; i < _stream_compute.size(); i++) {
    //   CHECK_CUDA(
    //       cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
    //   CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
    // }
    // CHECK_CUDA(cudaEventRecord(_stop_send, (cudaStream_t)_stream_send));
    // CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_send, 0));
    // CHECK_CUDA(cudaEventRecord(_stop_recv, (cudaStream_t)_stream_recv));
    // CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_recv, 0));
    // at::cuda::setCurrentCUDAStream(stream_main);

    // return D;
  }  // split_overlap_ag

};

}