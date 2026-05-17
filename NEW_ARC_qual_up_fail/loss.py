import torch
import torch.nn as nn
import torch.nn.functional as F

class IntensityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        
    def forward(self, I_fusion, I_org):
        return self.l1(I_fusion, I_org)

class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel-like kernels
        g_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        g_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        
        # Prepare for depthwise conv
        # Shape: [out_channels, in_channels / groups, kernel_size, kernel_size]
        # in_channels=3, out_channels=3, groups=3
        # Weight shape must be [3, 1, 3, 3]
        weight_x = g_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        weight_y = g_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)
        self.l1 = nn.L1Loss()
        
    def _compute_gradient(self, img):
        grad_x = F.conv2d(img, self.weight_x, groups=3, padding=1)
        grad_y = F.conv2d(img, self.weight_y, groups=3, padding=1)
        return torch.abs(grad_x) + torch.abs(grad_y)
        
    def forward(self, I_fusion, I_org):
        grad_fusion = self._compute_gradient(I_fusion)
        grad_org = self._compute_gradient(I_org)
        return self.l1(grad_fusion, grad_org)

class TotalLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=20.0): # beta 기본값 20.0으로 상향
        super().__init__()
        self.intensity_loss = IntensityLoss()
        self.gradient_loss = GradientLoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, I_fusion, I_org):
        l_int = self.intensity_loss(I_fusion, I_org)
        l_grad = self.gradient_loss(I_fusion, I_org)
        
        # [신규] Texture Loss (분산 비교를 통한 질감 보존)
        mu_x = F.avg_pool2d(I_fusion, 3, 1, 1)
        mu_y = F.avg_pool2d(I_org, 3, 1, 1)
        var_x = F.avg_pool2d(I_fusion**2, 3, 1, 1) - mu_x**2
        var_y = F.avg_pool2d(I_org**2, 3, 1, 1) - mu_y**2
        l_texture = torch.mean(torch.abs(var_x - var_y))
        
        # 질감(Texture) Loss 비중을 10.0으로 강력하게 부여
        l_total = self.alpha * l_int + self.beta * l_grad + 10.0 * l_texture
        
        # 반환값에 l_texture 추가 (총 4개 반환)
        return l_total, l_int, l_grad, l_texture

if __name__ == '__main__':
    criterion = TotalLoss()
    I_fusion = torch.randn(8, 3, 400, 600)
    I_org = torch.randn(8, 3, 400, 600)
    loss, l_int, l_grad, l_texture = criterion(I_fusion, I_org)
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Intensity Loss: {l_int.item():.4f}")
    print(f"Gradient Loss: {l_grad.item():.4f}")
    print(f"Texture Loss: {l_texture.item():.4f}")
    
    # Check depthwise conv shapes
    print(f"weight_x shape: {criterion.gradient_loss.weight_x.shape}")
