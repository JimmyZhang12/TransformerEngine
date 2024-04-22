import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
import transformer_engine_extensions as tex

import torch, time
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
wsize = comm.Get_size()

torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend='nccl', world_size=wsize, rank=rank)

# fp8_dtype = tex.DType.kFloat8E4M3
# recipe = transformer_engine.common.recipe.DelayedScaling(
#     fp8_format=transformer_engine.common.recipe.Format.E4M3,
# )
# with te.fp8_autocast(enabled=False, fp8_recipe=recipe):
#     te.module.base.initialize_ub(
#         shape=[2048,4096],
#         tp_size=wsize,
#     )
#     module = te.Linear(
#         in_features=32, 
#         out_features=32, 
#         tp_size=wsize,
#         ub_overlap_rs=True,
#         ub_overlap_ag=True,
#         ub_name='proj')
#     module.set_tensor_parallel_group(torch.distributed.group.WORLD)

#     _ = module(torch.zeros([8, 32], device="cuda"))
def test_baseline_ag_gemm(torch_dtype):
    input_features = 8192
    output_features = 8192*3
    sequence_len = 4096
    nsplit = 4
    tpsize = 4

    b = torch.rand([sequence_len//tpsize, input_features], dtype=torch_dtype).cuda()
    a = torch.rand([output_features//tpsize, input_features], dtype=torch_dtype).cuda()
    b_gathered = torch.zeros([sequence_len, input_features], dtype=torch_dtype).cuda()

    iters = 5
    for i in range(iters):
        if i == 1:
            torch.cuda.synchronize()
            tic = time.time()

        torch.distributed.all_gather_into_tensor(b_gathered, b)
        d = torch.matmul(b_gathered, a.t())

    torch.cuda.synchronize()
    toc = time.time()
    ttime = toc -tic
    avgtime = ttime/(iters-1)
    if torch.cuda.current_device() == 0:
        print(f"AVG TIME {avgtime}")

def test_nccl_ring_ex(torch_dtype):
    # A(outf, inpf)^T X B(seq, inpf)

    if torch_dtype == torch.float:
        te_type = tex.DType.kFloat32
    elif torch_dtype == torch.bfloat16:
        te_type = tex.DType.kBFloat16


    input_features = 8192
    output_features = 8192*3
    sequence_len = 4096
    nsplit = 4
    tpsize = 4

    A_chunk_dim = [output_features//tpsize,input_features]
    B_chunk_dim = [sequence_len//nsplit,input_features]
    D_chunk_dim = [sequence_len//nsplit,output_features//tpsize]

    B_numel = sequence_len//nsplit*input_features
    print(f"{B_chunk_dim} * {A_chunk_dim}^T")
    nccl_gemm_overlap = tex.NcclCommOverlap(B_numel, True, True)

    #emulate proj fprop
    A = torch.zeros(A_chunk_dim, dtype=torch.float).cuda()
    A_scale_inverse = torch.Tensor()
    A_fp8_tensor = -1
    A_type = te_type
    transa = True

    B = torch.zeros(B_chunk_dim, dtype=torch.float).cuda()
    B_scale_inverse = torch.Tensor()
    B_fp8_tensor = -1
    B_type = te_type
    transb = False

    D = torch.zeros(D_chunk_dim, dtype=torch.float).cuda()
    D_scale = torch.Tensor()
    D_type= te_type
    D_amax = torch.Tensor()

    bias = torch.Tensor()
    bias_type = te_type

    pre_gelu_out = torch.Tensor()
    grad = False

    workspace = torch.zeros([12*1024*1024],dtype=torch.uint8).cuda()
    workspaceSize = workspace.shape[0]

    accumulate=False
    use_split_accumulator = False
    comm_type = 0
    rs_output = torch.zeros([2,32], dtype=torch.float32).cuda()

    debug_print=False

    iters = 5
    for i in range(iters):
        if i == 1:
            torch.cuda.synchronize()
            tic = time.time()

        nccl_gemm_overlap.ring_exchange(
            A, A_scale_inverse,  A_fp8_tensor,
            A_type, transa,
            B, B_scale_inverse,  B_fp8_tensor, 
            B_type, transb, 
            D, D_scale, D_type,
            D_amax, bias, bias_type,
            pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, comm_type, rs_output,
            debug_print
        )
    torch.cuda.synchronize()
    toc = time.time()
    ttime = toc -tic
    avgtime = ttime/(iters-1)
    if torch.cuda.current_device() == 0:
        print(f"AVG TIME {avgtime}")

if __name__ == "__main__":
    torch.cuda.cudart().cudaProfilerStart()
    test_baseline_ag_gemm(torch.bfloat16)
    test_nccl_ring_ex(torch.bfloat16)
    torch.cuda.cudart().cudaProfilerStop()
