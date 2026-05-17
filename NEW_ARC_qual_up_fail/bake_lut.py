
import torch
import os
from model import SeAFusion

def bake():
    # 최적 모델 경로 설정 (18번 에폭)
    model_path = 'best_epoch_18_val_loss_1.7656.pt' 
    
    if not os.path.exists(model_path):
        print(f"Error: {model_path} 파일을 찾을 수 없습니다.")
        return

    # 모델 로드
    model = SeAFusion()
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.cuda()
    model.eval()

    print(f"Loading model from {model_path}...")

    # 3D LUT (17x17x17) 생성
    grid_size = 17
    x = torch.linspace(0, 1, grid_size)
    grid_r, grid_g, grid_b = torch.meshgrid(x, x, x, indexing='ij')
    input_rgb = torch.stack([grid_r, grid_g, grid_b], dim=-1).view(-1, 3).cuda()

    luts = []
    with torch.no_grad():
        for bin_val in [0.0, 1.0]:
            bin_tensor = torch.full((input_rgb.shape[0], 1), bin_val).cuda()
            fusion_input = torch.cat([input_rgb, bin_tensor], dim=1).view(-1, 4, 1, 1)
            out = model.nlut_fusion(fusion_input) 
            luts.append(out.view(grid_size, grid_size, grid_size, 3).permute(3, 0, 1, 2))

    full_lut = torch.stack(luts) 
    torch.save(full_lut, 'nlut_baked.pt')
    print(f"Successfully baked LUT! Saved as 'nlut_baked.pt'")

if __name__ == '__main__':
    bake()
