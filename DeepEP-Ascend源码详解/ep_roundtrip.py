# Linux Ascend NPU example; syntax checked, not device-tested.
import os
import torch
import torch.distributed as dist
import torch_npu
from deep_ep import EPBuffer

local_rank = int(os.environ['LOCAL_RANK'])
torch.npu.set_device(local_rank)
dist.init_process_group(backend='hccl')
group = dist.group.WORLD
P = dist.get_world_size()
T, M, H, K, E = 256, 256, 7168, 2, P * 4
device = torch.device(f'npu:{local_rank}')
x = torch.randn(T, H, dtype=torch.bfloat16, device=device)
idx = torch.topk(torch.rand(T, E, device=device), K, dim=1).indices.contiguous()
w = torch.full((T, K), 1.0 / K, dtype=torch.float32, device=device)
buf = EPBuffer(group, num_max_tokens_per_rank=M, hidden=H, num_topk=K,
               explicitly_destroy=True)
pending = buf.dispatch(x, topk_idx=idx, topk_weights=w, num_experts=E,
                       expert_alignment=128, do_expand=True, do_zero_padding=True,
                       async_with_compute_stream=True, defer_epilogue=True)
rx, _, rw, handle = pending.wait()
# 用恒等专家演示；真实模型在这里执行分组专家计算，并应用一次路由权重。
expert_out = (rx.float() * rw[:, None]).to(torch.bfloat16).contiguous()
y, _ = buf.combine(expert_out, handle=handle,
                   async_with_compute_stream=True, defer_epilogue=True).wait()
torch.npu.synchronize()
print(dist.get_rank(), (y.float() - x.float()).abs().max().item())
buf.destroy()
dist.destroy_process_group()
