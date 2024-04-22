/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include <nccl.h>
#include <mpi.h>

#include "comm_gemm_overlap_nccl.h"
#include "extensions.h"


nccl_ubuf::NcclCommOverlap::NcclCommOverlap(int m, bool userbuffers){
  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  printf("HelloWorld- myrank:%d ranks: %d\n", myrank, nranks);
  cudaSetDevice(myrank);

  ncclUniqueId id;
  ncclGetUniqueId(&id);

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

  size_t size = (1024 * 1024 * 1024); //1GB
  
  NCCLCHECK(ncclMemAlloc(&sendbuf, size)); // Allocate memory on GPU and store its address in devPtr
  NCCLCHECK(ncclMemAlloc(&recvbuf, size)); // Allocate memory on GPU and store its address in devPtr
  
  dim = m;

  _userbuffers = userbuffers;
  if (_userbuffers){
    NCCLCHECK(ncclCommRegister(comm, sendbuf, size, &sendRegHandle));
    NCCLCHECK(ncclCommRegister(comm, recvbuf, size, &recvRegHandle));
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  _math_sms = prop.multiProcessorCount - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);

  _stream_comm = at::cuda::getStreamFromPool(true); 

  cudaEventCreateWithFlags(&_start_compute, 0);
  cudaEventCreateWithFlags(&_stop_compute, 0);
  cudaEventCreateWithFlags(&_stop_comm, 0);

  at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
  for (int i = 0; i < nranks; i++) {
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
    _stream_compute.push_back(
        at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
  }

}

void nccl_ubuf::NcclCommOverlap::RingExchange(
  at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
  transformer_engine::DType A_type, bool transa,
  at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, 
  transformer_engine::DType B_type, bool transb, 
  at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
  at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
  at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
  bool accumulate, bool use_split_accumulator, int comm_type, at::Tensor rs_output,
  bool debug_print)
{  

  ncclDataType_t nccl_type; 
  if (A_type == transformer_engine::DType::kFloat32)
    ncclDataType_t nccl_type = ncclFloat;
  else if (A_type == transformer_engine::DType::kBFloat16)
    ncclDataType_t nccl_type = ncclBfloat16;


  int next_rank = (myrank + 1) % nranks;
  int prev_rank = (myrank - 1 + nranks) % nranks;

  // Catch up the default torch stream
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_compute, 0));


  int num_exchanges = nranks-1;
  for (int i=0; i<num_exchanges; i++){
    at::cuda::setCurrentCUDAStream(_stream_compute[i]);
    CHECK_CUDA(cudaStreamWaitEvent(_stream_compute[i], _stop_comm, 0));

    te_gemm(A, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb, D, D_scale,
            D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, _math_sms);

    ncclGroupStart();
    ncclSend(sendbuf, dim, nccl_type, next_rank, comm, _stream_comm.stream());
    ncclRecv(recvbuf, dim, nccl_type, prev_rank, comm, _stream_comm.stream());
    ncclGroupEnd();

  
    // next gemm chunk waits for this iteration's comm to finish
    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i+1], _stop_comm, 0));

  }

  at::cuda::setCurrentCUDAStream(_stream_compute[num_exchanges]);
  te_gemm(A, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb, D, D_scale,
          D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace, workspaceSize,
          accumulate, use_split_accumulator, _math_sms);

  // all streams must finish before returning to Pytorch main stream
  for (int i = 0; i < _stream_compute.size(); i++) {
    CHECK_CUDA(
        cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
  }
  CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
  at::cuda::setCurrentCUDAStream(stream_main);

}

void nccl_ubuf::NcclCommOverlap::print_tensor(torch::Tensor tensor){
  cudaDeviceSynchronize();
  for (int i=0; i<nranks; i++){
    if (i == myrank)
      std::cout << myrank << ":" << tensor << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
  }
}


nccl_ubuf::NcclCommOverlap::~NcclCommOverlap(){
  if (_userbuffers){
    NCCLCHECK(ncclCommDeregister(comm, sendRegHandle));
    NCCLCHECK(ncclCommDeregister(comm, recvRegHandle));
  }
  NCCLCHECK(ncclMemFree(sendbuf));
  NCCLCHECK(ncclMemFree(recvbuf));

  ncclCommDestroy(comm);
  MPI_Finalize(); 
}



