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
}

void nccl_ubuf::NcclCommOverlap::RingExchange(bool debug_print){
  auto options = torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  
  torch::Tensor my_tensor = torch::full({dim, dim}, myrank, options);

  if (debug_print)
    print_tensor(my_tensor);
  
  int num_exchanges = nranks*8;
  int next_rank = (myrank + 1) % nranks;
  int prev_rank = (myrank - 1 + nranks) % nranks;

  cudaDeviceSynchronize();
  MPI_Barrier(MPI_COMM_WORLD);


  for (int i=0; i<num_exchanges; i++){
    // size_t size = sizeof(float) * my_tensor.numel();
    // CHECK_CUDA(cudaMemcpyAsync(sendbuf, my_tensor.data_ptr(), 
    //   size, cudaMemcpyDeviceToDevice, 0));
    
    if (debug_print)
      printf("sending %d %d %d\n", myrank, next_rank, prev_rank);

    ncclGroupStart();
    ncclSend(sendbuf, my_tensor.numel(), ncclFloat, next_rank, comm, 0);
    ncclRecv(recvbuf, my_tensor.numel(), ncclFloat, prev_rank, comm, 0);
    ncclGroupEnd();


    if (debug_print){
      my_tensor = torch::from_blob(recvbuf, {dim, dim}, options);
      print_tensor(my_tensor);
    }
  }
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



