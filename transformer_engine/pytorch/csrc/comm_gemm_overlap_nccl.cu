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

#define MAX_THREADS 1024

nccl_ubuf::NcclCommOverlap::NcclCommOverlap(torch::Tensor sample, 
  int rank, int tp_size, int num_comm_sm, int num_max_streams, 
  bool set_sm_margin, bool aggregate2, bool is_reduce_scatter, bool atomic_gemm, bool userbuffers){

  _tp_size = tp_size;
  _tp_id = rank % tp_size;
  printf("HelloWorld- myrank:%d ranks: %d\n", _tp_id, _tp_size);

  ncclUniqueId id;
  if (_tp_id == 0){
    ncclGetUniqueId(&id);
    for (int i=1; i<_tp_size; i++){
      MPI_Send(&id, sizeof(id), MPI_BYTE, i, 0, MPI_COMM_WORLD);
    }
  }
  else{
    MPI_Status status;
    MPI_Recv(&id, sizeof(id), MPI_BYTE, 0, 0, MPI_COMM_WORLD, &status);
  }
  ncclCommInitRank(&comm, _tp_size, id, _tp_id);


  _self_chunk_id = _tp_id;
  _rank_round_tp = (rank / tp_size) * tp_size;
  _next_rank = (tp_size + rank + 1) % tp_size + _rank_round_tp;
  _prev_rank = (tp_size + rank + -1) % tp_size + _rank_round_tp;
  _ubuf_scale_inv_initialized = false;

  _userbuffers = userbuffers;
  _num_chunks = _tp_size;
  _atomic_gemm = false;
  _num_max_streams = num_max_streams;

  // Create workspace tensor with userbuffer
  int ubuf_bytes = sample.numel() * sample.element_size();
  int ubuf_chunk_bytes = ubuf_bytes / tp_size;
  int num_ubuf_chunks = tp_size;

  if (is_reduce_scatter) {
    // GEMM + RS overlap: Allocate `2 x tp_size - 1` buffers to hold recieved GEMM chunk
    // outputs for reduction at the end of the pipelining.
    ubuf_bytes = static_cast<int>(ubuf_chunk_bytes * (tp_size * 2 - 1));
    num_ubuf_chunks = static_cast<int>(tp_size * 2 - 1);
  }


  NCCLCHECK(ncclMemAlloc(&_ubuf_ptr, ubuf_bytes));
   if (_userbuffers){
    NCCLCHECK(ncclCommRegister(comm, _ubuf_ptr, ubuf_bytes, &_ubuf_handle));
  }

  _ubuf = torch::from_blob(
    _ubuf_ptr, {sample.size(0) / _tp_size * num_ubuf_chunks, sample.size(1)}, sample.options());

  // Create tensor chunks for easy management
  char *ubuf_byte_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  printf("ubuf_bytes %d, num_ubuf_chunks %d, _ubuf shape [%d, %d] \n", ubuf_bytes, num_ubuf_chunks, sample.size(0) / tp_size, sample.size(1));

  for (int i = 0; i < num_ubuf_chunks; i++) {
    torch::Tensor ubuf_chunk = torch::from_blob(
        ubuf_byte_ptr, {sample.size(0) / tp_size, sample.size(1)}, sample.options());
    _ubufs.push_back(ubuf_chunk);
    ubuf_byte_ptr += ubuf_chunk_bytes;
  }

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  _math_sms = prop.multiProcessorCount - transformer_engine::getenv<int>("NVTE_EXT_MARGIN_SM", 0);
  _stream_comm = at::cuda::getStreamFromPool(true); 

  printf("_num_max_streams %d _num_chuncks %d\n", _num_max_streams, _num_chunks );


  at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
  for (int i = 0; i < std::min(_num_max_streams, _num_chunks); i++) {
    cudaStream_t stream;
    cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, -1);
    _stream_compute.push_back(
        at::cuda::getStreamFromExternal(stream, stream_main.device_index()));
  }

  cudaEventCreateWithFlags(&_start_compute, 0);
  cudaEventCreateWithFlags(&_stop_compute, 0);
  cudaEventCreateWithFlags(&_stop_comm, 0);
  cudaEventCreateWithFlags(&_start_comm, 0);

}

  /*
  ** Bulk GEMM + COMM
  ** This function assumes the communication input is pre-copied to _ubuf
  */
  std::vector<at::Tensor>
  nccl_ubuf::NcclCommOverlap::bulk_overlap(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
               transformer_engine::DType A_type, bool transa, at::Tensor B,
               at::Tensor B_scale_inverse, int64_t B_fp8_tensor, transformer_engine::DType B_type,
               bool transb, at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
               at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
               at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
               bool accumulate, bool use_split_accumulator, int comm_type, at::Tensor rs_output) {

    // TODO
    ncclDataType_t nccl_type; 
    if (_ubuf.element_size() == 4)
      nccl_type = ncclFloat;
    else if (_ubuf.element_size() == 2)
      nccl_type = ncclBfloat16;
    else if (_ubuf.element_size() == 1)
      nccl_type = ncclUint8;

    COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);

    // Catch up the default torch stream
    at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
    CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t)stream_main));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

    // Communication: AG and RS
    int sendcount = _ubufs[0].numel();
    if (_comm_type == COMM_TYPE::AG) {
      int ubuf_chunk_bytes = _ubufs[0].numel()*_ubufs[0].element_size();
      void* recvbuff = _ubufs[0].data_ptr();
      void* sendbuff = recvbuff + _tp_id*ubuf_chunk_bytes;
      NCCLCHECK(ncclAllGather(sendbuff, recvbuff, _ubufs[0].numel(), nccl_type, comm, _stream_comm));

    } else if (_comm_type == COMM_TYPE::RS) {//No NCCL FP8 support so RS using ring exchange with explicit reduction
      for (int i=0; i<_tp_size; i++){
        printf("bulkoverap_rs: _tp_id %d i %d send_chunk_id %d recv_chunk_id %d do_gelu %d\n", 
            _tp_id, i, _ubufs[_tp_id - i], _ubufs[_tp_id - i - 1]);
        ncclGroupStart();
        ncclSend(_ubufs[_tp_id - i].data_ptr(), sendcount, nccl_type, _next_rank, comm, _stream_comm.stream());
        ncclRecv(_ubufs[_tp_id - i - 1].data_ptr(), sendcount, nccl_type, _prev_rank, comm, _stream_comm.stream());
        ncclGroupEnd();
      }

    } else {
      NVTE_ERROR("Not supported communication type.");
    }

    if (A_scale_inverse.numel())
      A_scale_inverse = A_scale_inverse[A_fp8_tensor];

    if (B_scale_inverse.numel())
      B_scale_inverse = B_scale_inverse[B_fp8_tensor];

    assert(pre_gelu_out.numel() == 0);
    te_gemm(A, A_scale_inverse, A_type, transa, B, B_scale_inverse, B_type, transb, D, D_scale,
            D_type, D_amax, bias, bias_type, pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, _math_sms);

    CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));

    if (_comm_type == COMM_TYPE::RS){      // Reduce GEMM output chunks
      char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[0].data_ptr());
      if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
        assert(_ubuf_scale_inv_initialized);
        char *rs_output_ptr = reinterpret_cast<char *>(_ubufs[_tp_size].data_ptr());
        float *scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
        reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, scale_inv_ptr,
                                _tp_size, _ubufs[0].numel(), (cudaStream_t) stream_main);
      } else {
        torch::Tensor rs_output = torch::from_blob(
          _ubufs[_tp_size].data_ptr(), {_ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
        torch::Tensor reduce_buf = torch::from_blob(
          reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
        torch::sum_out(rs_output, reduce_buf, 0);
      }
      return {D, rs_output};
    }
    else{
      return {D, _ubuf};
    }


  }  // bulk_overlap



torch::Tensor nccl_ubuf::NcclCommOverlap::split_overlap_ag_p2p(
  at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
  transformer_engine::DType A_type, bool transa,
  at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, 
  transformer_engine::DType B_type, bool transb, 
  at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
  at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
  at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
  bool accumulate, bool use_split_accumulator, at::Tensor B_copy)
{  
  ncclDataType_t nccl_type; 
  if (A_type == transformer_engine::DType::kFloat32)
    nccl_type = ncclFloat;
  else if (A_type == transformer_engine::DType::kBFloat16)
    nccl_type = ncclBfloat16;
  else if (A_type == transformer_engine::DType::kFloat8E4M3)
    nccl_type = ncclUint8;

  // Get GEMM comm_bytesensions between TN and NN input layouts
  const int m = (transa) ? A.size(0) : A.size(1);
  const int k = (transa) ? A.size(1) : A.size(0);
  const int n = (transb) ? B.size(1) : B.size(0);
  const int n_chunk = _ubufs[0].size(0);


  // Get communication and GEMM output chunk sizes
  const bool do_gelu = pre_gelu_out.numel() > 0;
  const int output_chunk_bytes = (do_gelu
                                  ? (n_chunk * m) * D.element_size()
                                  : (n_chunk * m) * HALF_BYTES);
  const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out.element_size() : 0;
  
  // Get output and workspace data pointers
  char *output_ptr = reinterpret_cast<char *>(D.data_ptr());
  char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out.data_ptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
  char *input_b_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int workspace_size_chunk = workspaceSize / _stream_compute.size();
  int comm_bytes = B.element_size() * n_chunk * k;

  if (A_scale_inverse.numel())
    A_scale_inverse = A_scale_inverse[A_fp8_tensor];

  if (B_scale_inverse.numel())
    B_scale_inverse = B_scale_inverse[B_fp8_tensor];

  printf(
    "m %d, n %d, k %d, n_chunk %d, _num_chunks %d comm_bytes %d element_size %d \n", \
    m,n,k,n_chunk, _num_chunks, comm_bytes, B.element_size(), do_gelu);

  // Catch up the default torch stream
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t)stream_main));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));
  CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[0], _start_compute, 0));

  for (int i=0; i<_num_chunks; i++){
    int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
    int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;

    printf("_tp_id %d i %d send_chunk_id %d recv_chunk_id %d do_gelu %d\n", 
        _tp_id, i, send_chunk_id, recv_chunk_id, do_gelu);

    if (i < _num_chunks - 1){
      ncclGroupStart();
      ncclSend(input_b_ptr + send_offset, comm_bytes, nccl_type, _next_rank, comm, _stream_comm.stream());
      ncclRecv(input_b_ptr + recv_offset, comm_bytes, nccl_type, _prev_rank, comm, _stream_comm.stream());
      ncclGroupEnd();

      // next gemm chunk waits for this iteration's comm to finish
      CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i+1], _stop_comm, 0));
    }

    // GEMM
    at::cuda::setCurrentCUDAStream(_stream_compute[i]);

    printf("_tp_id %d i %d send_chunk_id %d output_chunk_bytes %d\n", _tp_id, i , send_chunk_id, output_chunk_bytes);
    torch::Tensor output_chunk = torch::from_blob(
      output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk, m}, D.options());

    torch::Tensor input_b_chunk = torch::from_blob(
      input_b_ptr + send_offset, {n_chunk, k}, B.options());
    torch::Tensor workspace_chunk =
        torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                          {workspace_size_chunk}, workspace.options());
    if (do_gelu) {
      pre_gelu_out = torch::from_blob(
          pre_gelu_out_ptr + (send_chunk_id * aux_chunk_bytes),
          {n_chunk, m},
          pre_gelu_out.options());
    }

    te_gemm(A, A_scale_inverse, A_type, transa, 
            input_b_chunk, B_scale_inverse, B_type, transb, 
            output_chunk, D_scale, D_type, D_amax, 
            bias, bias_type, pre_gelu_out, grad, workspace_chunk, workspace_size_chunk,
            accumulate, use_split_accumulator, _math_sms);
  }

  // all streams must finish before returning to Pytorch main stream
  for (int i = 0; i < _stream_compute.size(); i++) {
    CHECK_CUDA(
        cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
  }
  CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
  at::cuda::setCurrentCUDAStream(stream_main);

  return D;


}

/*
** Split ReduceScatter + GEMM using P2P communication
*/
void nccl_ubuf::NcclCommOverlap::split_overlap_rs_p2p(at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
                      transformer_engine::DType A_type, bool transa, at::Tensor B,
                      at::Tensor B_scale_inverse, int64_t B_fp8_tensor,
                      transformer_engine::DType B_type, bool transb, at::Tensor D,
                      at::Tensor D_scale, transformer_engine::DType D_type, at::Tensor D_amax,
                      at::Tensor bias, transformer_engine::DType bias_type,
                      at::Tensor pre_gelu_out, bool grad, at::Tensor workspace,
                      size_t workspaceSize, bool accumulate, bool use_split_accumulator,
                      at::Tensor rs_output) {

  ncclDataType_t nccl_type; 
  if (D_type == transformer_engine::DType::kFloat32)
    nccl_type = ncclFloat;
  else if (D_type == transformer_engine::DType::kBFloat16)
    nccl_type = ncclBfloat16;
  else if (D_type == transformer_engine::DType::kFloat8E4M3)
    nccl_type = ncclUint8;

  int k = A.size(1);
  int n = B.size(0);

  // Get communication and GEMM input chunk sizes
  int n_chunk = n / _tp_size;
  const int comm_count = _ubufs[0].numel();
  const int comm_bytes = _ubufs[0].numel() * _ubufs[0].element_size();
  const int input_b_chunk_bytes = n_chunk * k * B.element_size();

  // Get input and workspace data pointers
  char *input_b_ptr = reinterpret_cast<char *>(B.data_ptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
  int workspace_size_chunk = workspaceSize / _stream_compute.size();

  if (A_scale_inverse.numel())
    A_scale_inverse = A_scale_inverse[A_fp8_tensor];

  if (B_scale_inverse.numel())
    B_scale_inverse = B_scale_inverse[B_fp8_tensor];

  at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
  CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
  for (int i = 0; i < _stream_compute.size(); i++) {
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_compute[i], _start_compute, 0));
  }
  CHECK_CUDA(cudaEventRecord(_start_comm, (cudaStream_t)stream_main));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_comm, 0));

  printf("input_b_chunk_bytes %d %d %d\n", input_b_chunk_bytes, n_chunk, k);
  printf("_tp_id %d comm_bytes %d comm_count %d %d\n", _tp_id, comm_bytes, comm_count, nccl_type);
  printf("_stream_compute.size() %d tp_size %d\n", _stream_compute.size(), _tp_size);

  // GEMM and send/recv chunks
  for (int i = 0; i < _tp_size; i++) {
    int input_b_chunk_id = (_tp_id + i + 1) % _tp_size;
    char* input_b_chunk_ptr = input_b_ptr + (input_b_chunk_id * input_b_chunk_bytes);

    torch::Tensor input_b_chunk = torch::from_blob(input_b_chunk_ptr, {n_chunk, k}, B.options());
    // Store the last GEMM chunk output to the recieve buffer.
    torch::Tensor workspace_chunk = torch::from_blob(
        workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
        {workspace_size_chunk}, workspace.options());

    if (i > 0) {
 
      // P2P communication chunk
      int send_rank = (_tp_id + i) % _tp_size + _rank_round_tp;
      int recv_rank = (_tp_size + _tp_id - i) % _tp_size + _rank_round_tp;

      CHECK_CUDA(cudaEventRecord(
          _start_comm, (cudaStream_t) _stream_compute[(i - 1) % _stream_compute.size()]));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t) _stream_comm, _start_comm, 0));
      
      ncclGroupStart();
      ncclSend(_ubuf[i - 1].data_ptr(), comm_count, nccl_type, _next_rank, comm, _stream_comm.stream());
      ncclRecv(_ubuf[i + _tp_size-1].data_ptr(), comm_count, nccl_type, _prev_rank, comm, _stream_comm.stream());
      ncclGroupEnd();
    }

    at::cuda::setCurrentCUDAStream(_stream_compute[i % _stream_compute.size()]);
    te_gemm(A, A_scale_inverse, A_type, transa, input_b_chunk, B_scale_inverse, B_type, transb,
            _ubufs[i], D_scale, D_type, D_amax, bias, bias_type, pre_gelu_out, grad,
            workspace_chunk, workspace_size_chunk, accumulate, use_split_accumulator,
            _math_sms);


  }

  // all streams must finish before returning to Pytorch main stream
  for (int i = 0; i < _stream_compute.size(); i++) {
    CHECK_CUDA(
        cudaEventRecord(_stop_compute, (cudaStream_t)_stream_compute[i]));
    CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_compute, 0));
  }
  CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)stream_main, _stop_comm, 0));
  at::cuda::setCurrentCUDAStream(stream_main);

  // Reduce GEMM output chunks
  char *reduce_buf_ptr = reinterpret_cast<char *>(_ubufs[_tp_size - 1].data_ptr());
  if (_ubuf.element_size() == 1 && rs_output.element_size() == 2) {
    assert(_ubuf_scale_inv_initialized);
    float *d_scale_inv_ptr = reinterpret_cast<float *>(_ubuf_scale_inv.data_ptr());
    char *rs_output_ptr = reinterpret_cast<char *>(rs_output.data_ptr());
    reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(reduce_buf_ptr, rs_output_ptr, d_scale_inv_ptr,
                            _tp_size, _ubufs[0].numel(), (cudaStream_t) stream_main);
  } else {
    torch::Tensor reduce_buf = torch::from_blob(
      reduce_buf_ptr, {_tp_size, _ubufs[0].size(0), _ubufs[0].size(1)}, _ubuf.options());
    torch::sum_out(rs_output, reduce_buf, 0);
  }
  
}


void nccl_ubuf::NcclCommOverlap::playground(
  at::Tensor A, at::Tensor A_scale_inverse, int64_t A_fp8_tensor,
  transformer_engine::DType A_type, bool transa,
  at::Tensor B, at::Tensor B_scale_inverse, int64_t B_fp8_tensor, 
  transformer_engine::DType B_type, bool transb, 
  at::Tensor D, at::Tensor D_scale, transformer_engine::DType D_type,
  at::Tensor D_amax, at::Tensor bias, transformer_engine::DType bias_type,
  at::Tensor pre_gelu_out, bool grad, at::Tensor workspace, size_t workspaceSize,
  bool accumulate, bool use_split_accumulator, at::Tensor B_copy)
{  
  ncclDataType_t nccl_type; 
  if (D_type == transformer_engine::DType::kFloat32)
    nccl_type = ncclFloat;
  else if (D_type == transformer_engine::DType::kBFloat16)
    nccl_type = ncclBfloat16;
  else if (D_type == transformer_engine::DType::kFloat8E4M3)
    nccl_type = ncclUint8;

  // Get GEMM comm_bytesensions between TN and NN input layouts
  const int m = (transa) ? A.size(0) : A.size(1);
  const int k = (transa) ? A.size(1) : A.size(0);
  const int n = (transb) ? B.size(1) : B.size(0);
  const int n_chunk = (n * _tp_size) / _num_chunks;


  // Get communication and GEMM output chunk sizes
  const bool do_gelu = pre_gelu_out.numel() > 0;
  const int output_chunk_bytes = (do_gelu
                                  ? (n_chunk * m) * D.element_size()
                                  : (n_chunk * m) * HALF_BYTES);
  const int aux_chunk_bytes = do_gelu ? (n_chunk * m) * pre_gelu_out.element_size() : 0;
  
  // Get output and workspace data pointers
  char *output_ptr = reinterpret_cast<char *>(D.data_ptr());
  char *pre_gelu_out_ptr = reinterpret_cast<char *>(pre_gelu_out.data_ptr());
  char *workspace_ptr = reinterpret_cast<char *>(workspace.data_ptr());
  char *input_b_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  int workspace_size_chunk = workspaceSize / _stream_compute.size();
  int comm_bytes = B.element_size() * n_chunk * k;

  if (A_scale_inverse.numel())
    A_scale_inverse = A_scale_inverse[A_fp8_tensor];

  if (B_scale_inverse.numel())
    B_scale_inverse = B_scale_inverse[B_fp8_tensor];

  printf(
    "m %d, n %d, k %d, n_chunk %d, _num_chunks %d comm_bytes %d element_size %d \n", \
    m,n,k,n_chunk, _num_chunks, comm_bytes, B.element_size());

  // Catch up the default torch stream
  at::cuda::CUDAStream stream_main = at::cuda::getCurrentCUDAStream();
  CHECK_CUDA(cudaEventRecord(_start_compute, (cudaStream_t)stream_main));
  CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_comm, _start_compute, 0));

  for (int i=0; i<_num_chunks; i++){
    int send_chunk_id = (_tp_size + _tp_id - i) % _tp_size;
    int recv_chunk_id = (_tp_size + _tp_id - i - 1) % _tp_size;
    int send_offset = comm_bytes * send_chunk_id;
    int recv_offset = comm_bytes * recv_chunk_id;

    printf("_tp_id %d i %d send_chunk_id %d recv_chunk_id %d\n", _tp_id, i, send_chunk_id, recv_chunk_id);

    if (i < _num_chunks - 1){
      ncclGroupStart();
      ncclSend(input_b_ptr + send_offset, comm_bytes, nccl_type, _next_rank, comm, _stream_comm.stream());
      ncclRecv(input_b_ptr + recv_offset, comm_bytes, nccl_type, _prev_rank, comm, _stream_comm.stream());
      ncclGroupEnd();

      // next gemm chunk waits for this iteration's comm to finish
      CHECK_CUDA(cudaEventRecord(_stop_comm, (cudaStream_t)_stream_comm));
      CHECK_CUDA(cudaStreamWaitEvent((cudaStream_t)_stream_compute[i+1], _stop_comm, 0));
    }

    // GEMM
    at::cuda::setCurrentCUDAStream(_stream_compute[i]);

    torch::Tensor output_chunk = torch::from_blob(
      output_ptr + (send_chunk_id * output_chunk_bytes), {n_chunk, m}, D.options());

    printf("inp %d %d %d\n", _tp_id, i , (send_chunk_id * output_chunk_bytes));
    torch::Tensor input_b_chunk = torch::from_blob(
      input_b_ptr + send_offset, {n_chunk, k}, B.options());
    torch::Tensor workspace_chunk =
        torch::from_blob(workspace_ptr + (i % _stream_compute.size()) * workspace_size_chunk,
                          {workspace_size_chunk}, workspace.options());


    te_gemm(A, A_scale_inverse, A_type, transa, 
            input_b_chunk, B_scale_inverse, B_type, transb, 
            output_chunk, D_scale, D_type, D_amax, 
            bias, bias_type, pre_gelu_out, grad, workspace_chunk, workspace_size_chunk,
            accumulate, use_split_accumulator, _math_sms);
  }

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


void nccl_ubuf::NcclCommOverlap::copy_input_to_ubuf(torch::Tensor input, bool chunk) {
  at::cuda::CUDAStream stream_main = at::cuda::getDefaultCUDAStream();
  if (chunk) {
    // Copy input to the target ubuf chunk by rank offset
    if (input.numel() != _ubufs[0].numel() || input.element_size() != _ubufs[0].element_size()) {
      NVTE_ERROR("input and ubuf size do not match! %d %d", input.numel(),  _ubufs[0].numel());
    }
    CHECK_CUDA(cudaMemcpyAsync(_ubufs[_tp_id].data_ptr(), input.data_ptr(),
                                input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream_main));
  } else {
    if (input.numel() != _ubuf.numel() || input.element_size() != _ubuf.element_size()) {
      NVTE_ERROR("input and ubuf size do not match!");
    }
    CHECK_CUDA(cudaMemcpyAsync(_ubuf.data_ptr(), input.data_ptr(),
                                input.numel() * input.element_size(), cudaMemcpyDeviceToDevice,
                                (cudaStream_t)stream_main));
  }
}


torch::Tensor nccl_ubuf::NcclCommOverlap::get_ubuf_output(int comm_type) {
  char *ubuf_wt_ptr = reinterpret_cast<char *>(_ubuf.data_ptr());
  COMM_TYPE _comm_type = static_cast<COMM_TYPE>(comm_type);
  if (_comm_type != COMM_TYPE::AG && _comm_type != COMM_TYPE::RS)
    NVTE_ERROR("Invalid comm_type");
  if (_comm_type == COMM_TYPE::RS)
    ubuf_wt_ptr += _ubuf.numel() / _tp_size * _self_chunk_id * _ubuf.element_size();
  int output_c_dim0 = (_comm_type == COMM_TYPE::AG) ? _ubuf.size(0) : _ubuf.size(0) / _tp_size;
  int output_c_dim1 = _ubuf.size(1);
  return torch::from_blob(ubuf_wt_ptr, {output_c_dim0, output_c_dim1}, _ubuf.options());
}

nccl_ubuf::NcclCommOverlap::~NcclCommOverlap(){
  if (_userbuffers){
    NCCLCHECK(ncclCommDeregister(comm, _ubuf_handle));
  }
  NCCLCHECK(ncclMemFree(_ubuf_ptr));
  ncclCommDestroy(comm);
  MPI_Finalize(); 
}


template <typename fp8type>
__global__ void __launch_bounds__(MAX_THREADS / 4)
reduce_fp8_in_bf16_out_cuda(void *inputs, void *output, const float *scale,
                            const int num_inputs, const int input_size) {
  const size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
  fp8type *inputs_fp8 = reinterpret_cast<fp8type *>(inputs);
  float accum_buf = static_cast<float>(inputs_fp8[tid]) * (*scale);
  #pragma unroll
  for (int i = 1; i < num_inputs; i++) {
    accum_buf += static_cast<float>(inputs_fp8[tid + input_size * i]) * (*scale);
  }
  half *output_half = reinterpret_cast<half *>(output);
  output_half[tid] = (half) accum_buf;
}

template <typename fp8type>
void nccl_ubuf::NcclCommOverlap::reduce_fp8_in_bf16_out(void *inputs, void *output, float *scale, int num_inputs,
                            int input_size, cudaStream_t stream) {
  size_t num_threads = MAX_THREADS / 4;
  size_t num_blocks = (input_size +num_threads - 1) / num_threads;
  dim3 block(num_threads);
  dim3 grid(num_blocks);
  reduce_fp8_in_bf16_out_cuda<fp8type><<<grid, block, 0, stream>>>(
    inputs, output, scale, num_inputs, input_size);
}

template void nccl_ubuf::NcclCommOverlap::reduce_fp8_in_bf16_out<__nv_fp8_e4m3>(
  void *inputs, void *output, float *scale, int num_inputs, int input_size, cudaStream_t stream);
template void nccl_ubuf::NcclCommOverlap::reduce_fp8_in_bf16_out<__nv_fp8_e5m2>(
  void *inputs, void *output, float *scale, int num_inputs, int input_size, cudaStream_t stream);
