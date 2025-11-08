import torch
from icecream import ic
import os

use_exp = True
init_phi = os.environ.get('MMEA_PHI', 'exp')
if init_phi == 'exp':
    use_exp = True
elif init_phi == 'res':
    use_exp = False


def calc(features, ori_out, irreps_in, irreps_out, adapter_tensor_head, adapter_scalar, adapter_scale):
    features_slices = [features[..., slice_obj] for slice_obj in irreps_in.slices()]

    norm_input = features_slices[0]
    
    adapter_hidden = adapter_tensor_head(norm_input)
    scale_params = adapter_scale(adapter_hidden)
    ic(adapter_hidden)
    
    modulated_out = ori_out.clone()

    out_slices = [modulated_out[..., slice_obj] for slice_obj in irreps_out.slices()]

    start_idx = 0
    updated_chunks = []
    for out, mul_ir in zip(out_slices, irreps_out):
        if mul_ir.ir.dim > 1:
            out = out.reshape(out.shape[0], -1, mul_ir.ir.dim).permute(0, 2, 1)
            scale = scale_params[:, start_idx:start_idx+mul_ir.mul].unsqueeze(1)
            if use_exp:
                # scale = torch.clamp(scale, max=10)
                out = out * torch.exp(scale)
            else:
                out = out * (1.0 + scale)
            out = out.permute(0, 2, 1).reshape(out.shape[0], -1)
            start_idx += mul_ir.mul
            updated_chunks.append(out)
        else:
            out = out + adapter_scalar(adapter_hidden)
            updated_chunks.append(out)
    
    modulated_out = torch.cat(updated_chunks, dim=-1)

    
    return modulated_out