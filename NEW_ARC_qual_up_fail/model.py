import torch
import torch.nn as nn
import torch.nn.functional as F

class LightConvNLUTFusion(nn.Module):
    """
    3. 신경망 기반 초경량 LUT (Neural Look-Up Table, NLUT)
    순수 1x1 Conv (Pointwise) 구조로 변경하여 추론 시 MAC 0 완벽 달성 가능.
    """
    def __init__(self, in_channels=4, hidden_dim=16):
        super().__init__()
        self.nlut = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 3, kernel_size=1), # RGB 출력
            nn.Sigmoid() # 출력값 범위를 [0, 1]로 안정화
        )
        
    def forward(self, fusion_input):
        return self.nlut(fusion_input)

class SeAFusion(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=16, tile_size=16):
        super().__init__()
        self.tile_size = tile_size
        self.nlut_fusion = LightConvNLUTFusion(in_channels=in_channels, hidden_dim=hidden_dim)

    def forward(self, general_img, I_bin, I_bin_mask, mode='train'):
        B, C, H, W = general_img.shape

        dilated_mask = F.max_pool2d(I_bin_mask, kernel_size=3, stride=1, padding=1)
        boundary_mask = dilated_mask - I_bin_mask 
        smooth_boundary = F.avg_pool2d(boundary_mask, kernel_size=3, stride=1, padding=1)

        fusion_input = torch.cat([general_img, I_bin], dim=1)
        fused_out = self.nlut_fusion(fusion_input)
        
        blended = (1.0 - smooth_boundary) * general_img + smooth_boundary * fused_out
        final_out = torch.where(I_bin_mask > 0, fused_out, blended)

        if mode == 'train':
            return final_out, boundary_mask
        else:
            return final_out

if __name__ == '__main__':
    model = SeAFusion()
    I_gen = torch.randn(8, 3, 400, 600)
    I_bin = torch.randn(8, 1, 400, 600)
    I_bin_mask = torch.zeros(8, 1, 400, 600)
    I_bin_mask[:, :, 200:250, 300:350] = 1.0 
    
    out, b_mask = model(I_gen, I_bin, I_bin_mask, mode='train')
    print(f"I_gen shape: {I_gen.shape}")
    print(f"I_bin shape: {I_bin.shape}")
    print(f"I_bin_mask shape: {I_bin_mask.shape}")
    print(f"I_fusion shape: {out.shape}")
    print(f"b_mask shape: {b_mask.shape}")
    
    out_inf = model(I_gen, I_bin, I_bin_mask, mode='inference')
    print(f"I_fusion (inf) shape: {out_inf.shape}")
