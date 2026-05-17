import torch
import numpy as np
import time
import os
from model import SeAFusion
from inference_lut import SeAFusionLUTInference

def profile():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Edge Environment Simulation (GPU Profiling on {device}) ---")
    
    # 1. Initialize models
    pt_model = SeAFusion().to(device)
    pt_model.eval()
    
    # Check if baked LUT exists, if not, skip LUT profiling or handle
    if not os.path.exists('nlut_baked.pt'):
        print("nlut_baked.pt not found. Please run bake_lut.py first.")
        return
        
    lut_model = SeAFusionLUTInference('nlut_baked.pt', device=str(device))
    
    # Save a dummy pt model to compare size
    torch.save(pt_model.state_dict(), 'dummy_weights.pt')
    pt_size = os.path.getsize('dummy_weights.pt') / 1024
    lut_size = os.path.getsize('nlut_baked.pt') / 1024
    
    print("\n[1] Model Size Comparison:")
    print(f"  PyTorch FP32 Weights: {pt_size:.2f} KB")
    print(f"  17-Grid 3D LUT Size:  {lut_size:.2f} KB")
    
    # Count MACs for the PyTorch NLUT
    total_params = sum(p.numel() for p in pt_model.parameters())
    print(f"  PyTorch Parameters:   {total_params:,}")
    print(f"  LUT Parameters:       0 (Trilinear Texture Interpolation)")
    
    # 2. Prepare Dummy Data
    B, C, H, W = 1, 3, 400, 600
    I_gen = torch.rand(1, 3, H, W, device=device)
    
    # I_bin은 0.0 또는 1.0의 값만 가지는 이진 마스크
    I_bin = torch.randint(0, 2, (1, 1, H, W), dtype=torch.float32, device=device)
    
    # 3. Correctness Verification
    with torch.no_grad():
        out_pt_tensor = pt_model(I_gen, I_bin, I_bin, mode='inference')
        
    out_lut_tensor, _ = lut_model.process(I_gen, I_bin, I_bin)
    
    # Calculate MAE in float [0, 1] scale then multiply by 255 for standard pixel scale
    diff = torch.abs(out_pt_tensor - out_lut_tensor)
    mae = torch.mean(diff).item() * 255.0
    max_diff = torch.max(diff).item() * 255.0
    
    print("\n[2] Correctness Verification (PyTorch Full vs PyTorch LUT):")
    print(f"  Mean Absolute Error (0-255 scale): {mae:.4f}")
    print(f"  Max Pixel Difference:              {max_diff:.4f}")
    
    # 4. FPS Measurement
    print("\n[3] FPS Profiling (100 iterations):")
    n_iters = 100
    
    # PyTorch FPS
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = pt_model(I_gen, I_bin, I_bin, mode='inference')
            if device.type == 'cuda':
                torch.cuda.synchronize()
    pt_time = (time.time() - start) / n_iters
    
    # LUT FPS
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_iters):
        _ = lut_model.process(I_gen, I_bin, I_bin)
        # process method handles synchronization
    lut_time = (time.time() - start) / n_iters
    
    print(f"  PyTorch FPS:           {1.0 / pt_time:.2f} (Time per frame: {pt_time*1000:.2f} ms)")
    print(f"  PyTorch LUT (3D) FPS:  {1.0 / lut_time:.2f} (Time per frame: {lut_time*1000:.2f} ms)")
    print(f"  -> Speedup:            {pt_time / lut_time:.2f}x")
    
if __name__ == '__main__':
    profile()
