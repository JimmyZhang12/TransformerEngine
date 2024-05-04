import torch

import transformer_engine.common.recipe
import transformer_engine.pytorch as te
from transformer_engine.pytorch.float8_tensor import Float8Tensor
import transformer_engine_extensions as tex

import torch, time, os
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
wsize = comm.Get_size()

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

torch.cuda.set_device(rank)
torch.distributed.init_process_group(backend='nccl', world_size=wsize, rank=rank)


class BasicTEMLP(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        activation_dtype = torch.bfloat16
        tp_size = wsize
        output_features = 8192
        input_features = 8192
        seqlen = 2048
        tp_group = torch.distributed.new_group()

        self.qkv = te.LayerNormLinear(
            in_features=input_features, 
            out_features=output_features, 
            tp_size=wsize,
            ub_overlap_ag=True,
            ub_name='qkv',
            tp_group=tp_group,
            sequence_parallel=True,
            bias=False,
            parallel_mode='column',
            params_dtype=activation_dtype   
        ) 

        self.proj = te.Linear(
            in_features=input_features, 
            out_features=output_features, 
            tp_size=wsize,
            ub_overlap_rs=True,
            ub_overlap_ag=True,
            ub_name='proj',
            tp_group=tp_group,
            sequence_parallel=True,
            bias=False,
            parallel_mode='row',
            params_dtype=activation_dtype   
        ) 


    def forward(self, x):
        x = self.qkv(x, is_first_microbatch=False)
        x = self.proj(x, is_first_microbatch=False)
        return x


def userbuffers():
        output_features = 8192
        input_features = 8192
        seqlen = 2048
        tp_size = wsize
        activation_dtype = torch.bfloat16


        ub_cfgs = {
            "qkv_fprop": {
                'method': "ring_exchange",
                'use_nccl': True,
                'use_nccl_userbuffer': False,
                'fp8_buf': True,
            },
            "proj_fprop": {
                'method': "ring_exchange",
                'use_nccl': True,
                'use_nccl_userbuffer': False,
                'fp8_buf': True,
            }
        }
        
        te.module.base.initialize_ub(
            shape=[seqlen,output_features],
            tp_size=tp_size,
            ub_cfgs=ub_cfgs,
            use_fp8=True,
            dtype=activation_dtype,
        )


        num_modules = 16
        modules = []
        with te.fp8_model_init(enabled=True):
            for _ in range(num_modules):
                modules.append(BasicTEMLP())

        recipe = transformer_engine.common.recipe.DelayedScaling(
            fp8_format=transformer_engine.common.recipe.Format.E4M3,
        )
        with te.fp8_autocast(enabled=True, fp8_recipe=recipe):
            x = torch.zeros([seqlen//tp_size, input_features], dtype=activation_dtype, device=torch.cuda.current_device())

            for mlp in modules:
                x = mlp(x)


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

def test_ag(
    input_features, 
    output_features, 
    sequence_len,
    cpsize,
    tpsize,
    torch_dtype):
    # A(outf, inpf)^T X B(seq, inpf)

    if torch_dtype == torch.float:
        te_type = tex.DType.kFloat32
    elif torch_dtype == torch.bfloat16:
        te_type = tex.DType.kBFloat16
    elif torch_dtype == torch.uint8:
        te_type = tex.DType.kFloat8E4M3

    gemm_chunks = tpsize

    A_chunk_dim = [output_features//tpsize,input_features]
    B_chunk_dim = [sequence_len//tpsize//cpsize,input_features]
    D_chunk_dim = [sequence_len//cpsize,output_features//tpsize]

    sample = torch.zeros([sequence_len//cpsize, input_features], dtype=torch_dtype).cuda()
    nccl_gemm_overlap = tex.NcclCommOverlap(sample, gemm_chunks, False, True)
    
    #emulate proj fprop
    A = torch.ones(A_chunk_dim, dtype=torch_dtype).cuda()
    A_scale_inverse = torch.Tensor([1,1,1]).cuda()
    A_fp8_tensor = tex.FP8FwdTensors.GEMM1_WEIGHT
    A_type = te_type
    transa = True

    B = torch.ones(B_chunk_dim, dtype=torch_dtype).cuda()
    B_scale_inverse = torch.Tensor([1,1,1]).cuda()
    B_fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
    B_type = te_type
    transb = False

    D = torch.zeros(D_chunk_dim, dtype=torch.bfloat16).cuda()
    D_scale = torch.Tensor()
    D_type = tex.DType.kBFloat16
    D_amax = torch.Tensor()

    bias = torch.Tensor()
    bias_type = te_type

    pre_gelu_out = torch.Tensor()
    grad = False

    workspace = torch.zeros([12*1024*1024],dtype=torch.uint8).cuda()
    workspaceSize = workspace.shape[0]

    accumulate=False
    use_split_accumulator = False
    rs_output = torch.zeros([2,32], dtype=torch.float32).cuda()

    debug_print=False

    iters = 5
    for i in range(iters):
        # if i == 1:
        #     torch.cuda.synchronize()
        #     tic = time.time()
        nccl_gemm_overlap.copy_input_to_ubuf(B, True)
        # if torch.cuda.current_device() == 0:
        #     import pdb
        #     pdb.set_trace()
        # torch.distributed.barrier()  

        out = nccl_gemm_overlap.split_overap_ag(
            A, A_scale_inverse,  A_fp8_tensor,
            A_type, transa,
            B, B_scale_inverse,  B_fp8_tensor, 
            B_type, transb, 
            D, D_scale, D_type,
            D_amax, bias, bias_type,
            pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, rs_output)
        torch.cuda.synchronize()

        # if torch.cuda.current_device() == 0:
        #     import pdb
        #     pdb.set_trace()
        # torch.distributed.barrier()    
    # fp8_gemm(
    #     A: torch.Tensor,
    #     A_scale_inv: torch.Tensor,
    #     A_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    #     A_dtype: tex.DType,
    #     B: torch.Tensor,
    #     B_scale_inv: torch.Tensor,
    #     B_fp8_tensor: Union[tex.FP8FwdTensors, tex.FP8BwdTensors],
    #     B_dtype: tex.DType,
    #     out_dtype: torch.dtype,
    #     workspace: torch.Tensor,
    #     gelu: bool = False,
    #     accumulate: bool = False,
    #     out: Optional[torch.Tensor] = None,
    #     out_index = None,
    #     fp8_meta_tensor: tex.FP8TensorMeta = None,
    #     bias: Optional[torch.Tensor] = None,
    #     use_bias: bool = False,
    #     use_split_accumulator: bool = False,
    #     D_dtype: Optional[tex.DType] = None,

    # toc = time.time()
    # ttime = toc -tic
    # avgtime = ttime/(iters-1)
    # if torch.cuda.current_device() == 0:
    #     print(f"AVG TIME {avgtime}")




def test_rs(
    input_features, 
    output_features, 
    sequence_len,
    cpsize,
    tpsize,
    torch_dtype):
    # A(outf, inpf)^T X B(seq, inpf)

    if torch_dtype == torch.float:
        te_type = tex.DType.kFloat32
    elif torch_dtype == torch.bfloat16:
        te_type = tex.DType.kBFloat16
    elif torch_dtype == torch.uint8:
        te_type = tex.DType.kFloat8E4M3

    gemm_chunks = tpsize

    A_chunk_dim = [output_features,input_features//tpsize]
    B_chunk_dim = [sequence_len//cpsize,input_features//tpsize]
    D_chunk_dim = [sequence_len//cpsize, output_features]

    sample = torch.zeros([sequence_len//cpsize * tpsize, output_features], dtype=torch_dtype).cuda()
    nccl_gemm_overlap = tex.NcclCommOverlap(
        sample, gemm_chunks, False, gemm_chunks, True)
    
    A = torch.ones(A_chunk_dim, dtype=torch_dtype).cuda()
    A_scale_inverse = torch.Tensor([1,1,1]).cuda()
    A_fp8_tensor = tex.FP8FwdTensors.GEMM1_WEIGHT
    A_type = te_type
    transa = True

    B = torch.ones(B_chunk_dim, dtype=torch_dtype).cuda()
    B_scale_inverse = torch.Tensor([1,1,1]).cuda()
    B_fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
    B_type = te_type
    transb = False

    D = torch.zeros(D_chunk_dim, dtype=torch.bfloat16).cuda()
    D_scale = torch.Tensor()
    D_type = tex.DType.kBFloat16
    D_amax = torch.Tensor()

    bias = torch.Tensor()
    bias_type = te_type

    pre_gelu_out = torch.Tensor()
    grad = False

    workspace = torch.zeros([64*1024*1024],dtype=torch.uint8).cuda()
    workspaceSize = workspace.shape[0]

    accumulate=False
    use_split_accumulator = False
    rs_output = torch.zeros([1024*8192], dtype=torch_dtype).cuda()

    debug_print=False

    iters = 5

    test_a = torch.zeros([32768, 32768], dtype=torch.float).cuda()
    test_b = torch.zeros([32768, 32768], dtype=torch.float).cuda()

    for i in range(iters):
        c = torch.add(test_a, test_b)

        out = nccl_gemm_overlap.split_overlap_rs(
            A, A_scale_inverse,  A_fp8_tensor,
            A_type, transa,
            B, B_scale_inverse,  B_fp8_tensor, 
            B_type, transb, 
            D, D_scale, D_type,
            D_amax, bias, bias_type,
            pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, rs_output)


def playground(
    input_features, 
    output_features, 
    sequence_len,
    cpsize,
    tpsize,
    torch_dtype):
    # A(outf, inpf)^T X B(seq, inpf)

    if torch_dtype == torch.float:
        te_type = tex.DType.kFloat32
    elif torch_dtype == torch.bfloat16:
        te_type = tex.DType.kBFloat16
    elif torch_dtype == torch.uint8:
        te_type = tex.DType.kFloat8E4M3


    A_chunk_dim = [output_features,input_features//tpsize]
    B_chunk_dim = [sequence_len//cpsize,input_features//tpsize]
    D_chunk_dim = [sequence_len//cpsize, output_features]

    gemm_chunks = tpsize
    sample = torch.zeros([sequence_len//cpsize * tpsize, output_features], dtype=torch_dtype).cuda()
    nccl_gemm_overlap = tex.NcclCommOverlap(sample, gemm_chunks, False, gemm_chunks)
    
    A = torch.ones(A_chunk_dim, dtype=torch_dtype).cuda()
    A_scale_inverse = torch.Tensor([1,1,1]).cuda()
    A_fp8_tensor = tex.FP8FwdTensors.GEMM1_WEIGHT
    A_type = te_type
    transa = True

    B = torch.ones(B_chunk_dim, dtype=torch_dtype).cuda()
    B_scale_inverse = torch.Tensor([1,1,1]).cuda()
    B_fp8_tensor = tex.FP8FwdTensors.GEMM1_INPUT
    B_type = te_type
    transb = False

    D = torch.zeros(D_chunk_dim, dtype=torch.bfloat16).cuda()
    D_scale = torch.Tensor()
    D_type = tex.DType.kBFloat16
    D_amax = torch.Tensor()

    bias = torch.Tensor()
    bias_type = te_type

    pre_gelu_out = torch.Tensor()
    grad = False

    workspace = torch.zeros([64*1024*1024],dtype=torch.uint8).cuda()
    workspaceSize = workspace.shape[0]

    accumulate=False
    use_split_accumulator = False
    rs_output = torch.zeros([1024*8192], dtype=torch_dtype).cuda()

    debug_print=False

    iters = 8
    for i in range(iters):
        # if i == 1:
        #     torch.cuda.synchronize()
        #     tic = time.time()
        out = nccl_gemm_overlap.playground(
            A, A_scale_inverse,  A_fp8_tensor,
            A_type, transa,
            B, B_scale_inverse,  B_fp8_tensor, 
            B_type, transb, 
            D, D_scale, D_type,
            D_amax, bias, bias_type,
            pre_gelu_out, grad, workspace, workspaceSize,
            accumulate, use_split_accumulator, rs_output)
        torch.cuda.synchronize()



if __name__ == "__main__":
    # userbuffers()
    torch.cuda.cudart().cudaProfilerStart()
    # test_baseline_ag_gemm(torch.bfloat16)
    # test_ag(
    #     input_features = 8192,
    #     output_features = 10240, #GQA (8192/64*(64+8+8)) 
    #     sequence_len = 4096,
    #     tpsize = 2,
    #     cpsize = 2,
    #     torch_dtype = torch.bfloat16)

    userbuffers()

    # test_rs(
    #     input_features = 8192,
    #     output_features = 8192, #FFN1 chunks gate plus fc1
    #     sequence_len = 4096,
    #     tpsize = 2,
    #     cpsize = 2,
    #     torch_dtype = torch.uint8)


    # test_ag(
    #     input_features = 8192,
    #     output_features = 28672*2, #FFN1 chunks gate plus fc1
    #     sequence_len = 4096,
    #     tpsize = 2,
    #     cpsize = 2,
    #     torch_dtype = torch.uint8)


    # playground(
    #     input_features = 8192,
    #     output_features = 28672*2, #FFN1 chunks gate plus fc1
    #     sequence_len = 4096,
    #     tpsize = 2,
    #     cpsize = 1,
    #     torch_dtype = torch.uint8)

    torch.cuda.cudart().cudaProfilerStop()
