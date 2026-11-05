# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

import torch
import torch_qaic
import torch_qaic.custom_ops as ops

device = "qaic:0"

n      = 1024  # must be a multiple of 64 (HVX vector width in fp16 elements)
x      = torch.ones(n, dtype=torch.float16).to(device)
out    = torch.zeros(n, dtype=torch.float16).to(device)
count  = torch.tensor([n], dtype=torch.int32).to(device)
scale  = torch.tensor([3.14], dtype=torch.float16).to(device)

print(f"input[:4]:  {x.cpu()[:4]}")
print(f"scale:      {scale.cpu().item():.4f}")

# Dispatch: 1 NSP, 4 threads
ops.scale_fp16[1, 4](x, out, count, scale)
torch.qaic.synchronize()

result   = out.cpu().float()
expected = float(torch.tensor(3.14, dtype=torch.float16))  # fp16-rounded reference
print(f"output[:4]: {out.cpu()[:4]}")
print(f"max error:  {(result - expected).abs().max().item():.6f}")
print("PASS" if (result - expected).abs().max().item() < 1e-3 else "FAIL")
