import torch
import torch.nn.functional as F
import time
import os

class SeAFusionLUTInference:
    def __init__(self, lut_path='nlut_baked.pt', device='cuda'):
        if not os.path.exists(lut_path):
            raise FileNotFoundError(f"LUT file not found: {lut_path}. Please run bake_lut.py first.")
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Loading LUT from {lut_path} onto {self.device}...")
        
        start_time = time.time()
        # lut shape: (2, 3, 17, 17, 17)
        self.lut = torch.load(lut_path, map_location=self.device)
        print(f"LUT loaded in {time.time() - start_time:.2f} seconds. Shape: {self.lut.shape}")
        
    def process(self, I_sat, I_bin, I_bin_mask):
        """
        I_sat: (1, 3, H, W) float32 tensor [0, 1]
        I_bin: (1, 1, H, W) float32 tensor [0, 1] representing binary mask
        I_bin_mask: (1, 1, H, W) float32 tensor [0, 1] saturation mask
        """
        start_time = time.time()
        
        B, C, H, W = I_sat.shape
        
        # 1. Prepare Grid for F.grid_sample
        # F.grid_sample expects grid in [-1, 1] range.
        # I_sat is [0, 1], so grid = I_sat * 2.0 - 1.0
        # The 3D LUT dimension order is R, G, B. 
        # grid_sample(D, H, W) expects (x, y, z) corresponding to (W, H, D).
        # So x=B, y=G, z=R
        r = I_sat[:, 0:1, :, :]
        g = I_sat[:, 1:2, :, :]
        b = I_sat[:, 2:3, :, :]
        
        # Shape: (1, 3, H, W) -> permute to (1, H, W, 3) for grid
        # Order: B, G, R (x, y, z)
        grid = torch.cat([b, g, r], dim=1).permute(0, 2, 3, 1) # (1, H, W, 3)
        grid = grid * 2.0 - 1.0
        grid = grid.unsqueeze(1) # (1, 1, H, W, 3)
        
        # Expand grid to batch size 2 (to query all 2 Bin LUTs simultaneously)
        grid = grid.expand(2, 1, H, W, 3)
        
        # 2. 3D LUT Trilinear Interpolation
        # self.lut shape: (2, 3, 17, 17, 17)
        # grid shape: (2, 1, H, W, 3)
        # output shape: (2, 3, 1, H, W) -> squeeze to (2, 3, H, W)
        fused_all = F.grid_sample(self.lut, grid, mode='bilinear', padding_mode='border', align_corners=True)
        fused_all = fused_all.squeeze(2) # (2, 3, H, W)
        
        # 3. Soft Blending (경계선 안티앨리어싱 및 질감 개선)
        # 이진 마스크(I_bin)를 3x3 평균 풀링으로 부드럽게 스무딩하여 Alpha Map 생성
        alpha_map = F.avg_pool2d(I_bin, kernel_size=3, stride=1, padding=1) # [1, 1, H, W]
        alpha_map = alpha_map.expand(1, 3, H, W) # [1, 3, H, W]
        
        # LUT 0번(짧은 노출/비번짐)과 1번(긴 노출/번짐) 결과 분리
        lut_out_0 = fused_all[0:1] # [1, 3, H, W]
        lut_out_1 = fused_all[1:2] # [1, 3, H, W]
        
        # Alpha Blending으로 두 결과 자연스럽게 혼합
        fused_out = (1.0 - alpha_map) * lut_out_0 + alpha_map * lut_out_1
        
        # 4. GPU Morphological Boundary Processing
        dilated_mask = F.max_pool2d(I_bin_mask, kernel_size=3, stride=1, padding=1)
        boundary_mask = dilated_mask - I_bin_mask
        smooth_boundary = F.avg_pool2d(boundary_mask, kernel_size=3, stride=1, padding=1)
        
        # 5. GPU Hardware Blending
        blended = (1.0 - smooth_boundary) * I_sat + smooth_boundary * fused_out
        final_out = torch.where(I_bin_mask > 0, fused_out, blended)
        
        # Wait for CUDA execution to finish to measure exact time
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        inference_time = time.time() - start_time
        return final_out, inference_time

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    print("Step 1. 테스트 환경 및 모델 준비")
    try:
        inferencer = SeAFusionLUTInference()
        device = inferencer.device
    except FileNotFoundError as e:
        print(e)
        exit(1)

    print("Step 2. 깨끗한 원본 로드 및 가짜 빛 번짐(포화) 생성")
    img_path = 'test.jpg'
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"'{img_path}' not found. Please provide a valid image.")
        
    img_np = cv2.imread(img_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (600, 400))
    I_ori = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0

    H, W = I_ori.shape[1], I_ori.shape[2]

    # L 값을 0.5 ~ 2.0 사이의 랜덤 값으로 뽑습니다.
    L = np.random.uniform(0.5, 2.0)

    # 랜덤한 중심점(cx, cy)
    cx = np.random.uniform(W / 2 - W * 0.05, W / 2 + W * 0.05)
    cy = np.random.uniform(H / 2 - H * 0.05, H / 2 + H * 0.05)

    y, x = torch.meshgrid(torch.arange(H, dtype=torch.float32), 
                          torch.arange(W, dtype=torch.float32), indexing='ij')

    # 거리 D
    D = torch.sqrt((x - cx)**2 + (y - cy)**2)

    # 방사형 밝기 마스크 M
    r = 300.0
    M = torch.clamp(1.0 - D / (r * L), min=0.0).unsqueeze(0) # [1, H, W]

    # I_ori + M * B_max * L 공식을 적용해 클리핑
    B_max = 150.0 / 255.0
    I_sat = torch.clamp(I_ori + M * B_max * L, min=0.0, max=1.0)

    print("Step 3. 추론용 마스크 추출 (포화 마스크 & 이진 마스크)")
    # 포화 마스크(I_bin_mask / M_sat) - 값이 1.0(또는 거의 1.0) 이상인 픽셀
    M_sat = (I_sat >= 0.99).any(dim=0, keepdim=True).float() # [1, H, W]
    I_bin_mask = M_sat

    # 이진 마스크(I_bin)
    I_gray = 0.299 * I_ori[0, :, :] + 0.587 * I_ori[1, :, :] + 0.114 * I_ori[2, :, :]
    I_gray = I_gray.unsqueeze(0) # [1, H, W]

    I_gray_np = (I_gray.squeeze(0).numpy() * 255.0).astype(np.uint8)
    otsu_thresh, _ = cv2.threshold(I_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    T_otsu = otsu_thresh / 255.0

    I_bin = ((I_bin_mask > 0) & (I_gray > T_otsu)).float()

    print("Step 4. LUT 추론 실행 및 결과 도출")
    # 배치 차원을 맞추고 GPU로 이동
    I_sat_batch = I_sat.unsqueeze(0).to(device)
    I_bin_batch = I_bin.unsqueeze(0).to(device)
    I_bin_mask_batch = I_bin_mask.unsqueeze(0).to(device)

    final_out, infer_time = inferencer.process(I_sat_batch, I_bin_batch, I_bin_mask_batch)
    print(f"Inference complete in {infer_time*1000:.2f} ms")

    # 결과 도출 및 나란히 비교
    final_out_cpu = final_out.squeeze(0).cpu().clamp(0, 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original (I_ori)")
    plt.imshow(I_ori.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("Degraded (I_sat)")
    plt.imshow(I_sat.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("Restored (final_out)")
    plt.imshow(final_out_cpu.permute(1, 2, 0).numpy())
    plt.axis('off')

    plt.tight_layout()
    res_path = 'inference_result.png'
    plt.savefig(res_path)
    print(f"Result saved to '{res_path}'")
