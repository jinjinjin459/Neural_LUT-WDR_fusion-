import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class SeAFusionDataset(Dataset):
    def __init__(self, image_dir=None, num_dummy=100):
        super().__init__()
        self.image_dir = image_dir
        self.image_paths = []
        self.num_dummy = num_dummy
        self.H = 400
        self.W = 600
        self.r = 300.0
        self.B_max = 150.0 / 255.0
        
        if self.image_dir and os.path.exists(self.image_dir):
            for fname in sorted(os.listdir(self.image_dir)):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    self.image_paths.append(os.path.join(self.image_dir, fname))
                    
        # Generate grid for distance computation
        y, x = torch.meshgrid(torch.arange(self.H, dtype=torch.float32), 
                              torch.arange(self.W, dtype=torch.float32), indexing='ij')
        self.grid_y = y
        self.grid_x = x

    def __len__(self):
        if len(self.image_paths) > 0:
            return len(self.image_paths)
        return self.num_dummy

    def __getitem__(self, idx):
        if len(self.image_paths) > 0:
            img_path = self.image_paths[idx]
            # Read and resize
            img_np = cv2.imread(img_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_np = cv2.resize(img_np, (self.W, self.H))
            # H, W, C -> C, H, W
            img_tensor = torch.from_numpy(img_np).float().permute(2, 0, 1) / 255.0
        else:
            # Dummy image data
            img_tensor = torch.rand(3, self.H, self.W, dtype=torch.float32)

        # I_org = img_tensor: [3, 400, 600]
        I_org = img_tensor

        # Generate I_gen
        # 중심(W/2, H/2)에서 ±5% 범위 내에서 랜덤하게 중심점 선택
        cx = np.random.uniform(self.W / 2 - self.W * 0.05, self.W / 2 + self.W * 0.05)
        cy = np.random.uniform(self.H / 2 - self.H * 0.05, self.H / 2 + self.H * 0.05)
        L = np.random.uniform(0.5, 2.0)
        
        # Distance tensor: D(x, y) = \sqrt{(x - cx)^2 + (y - cy)^2}
        D = torch.sqrt((self.grid_x - cx)**2 + (self.grid_y - cy)**2)
        
        # Brightness mask (1 channel): M = \max(0, 1 - \frac{D}{r \times L})
        M = torch.clamp(1.0 - D / (self.r * L), min=0.0)
        M = M.unsqueeze(0) # [1, 400, 600]
        
        # Synthesis and clipping: I_gen = \min(1.0, I_org + M \times B_max \times L)
        I_gen = torch.clamp(I_org + M * self.B_max * L, min=0.0, max=1.0)
        
        # Generate I_bin
        # I_gray (1 channel)
        # Using standard luminosity weights: 0.299*R + 0.587*G + 0.114*B
        I_gray = 0.299 * I_org[0, :, :] + 0.587 * I_org[1, :, :] + 0.114 * I_org[2, :, :]
        I_gray = I_gray.unsqueeze(0) # [1, 400, 600]
        
        # Otsu thresholding
        I_gray_np = (I_gray.squeeze(0).numpy() * 255.0).astype(np.uint8)
        otsu_thresh, _ = cv2.threshold(I_gray_np, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        T_otsu = otsu_thresh / 255.0
        
        # Saturation Mask: M_sat (any channel of I_gen is 1.0)
        # I_gen float precision issue could occur with exact 1.0, so check >= 1.0 or very close
        # torch.clamp guarantees max is 1.0
        M_sat = (I_gen >= 1.0).any(dim=0, keepdim=True) # [1, 400, 600]
        
        # I_bin = 1.0 if M_sat is True AND I_gray > T_otsu else 0.0
        I_bin = (M_sat & (I_gray > T_otsu)).float()
        
        return I_org, I_gen, I_bin

if __name__ == '__main__':
    # Test dataset output
    dataset = SeAFusionDataset(num_dummy=2)
    I_org, I_gen, I_bin = dataset[0]
    print(f"I_org shape: {I_org.shape}, dtype: {I_org.dtype}, range: [{I_org.min():.2f}, {I_org.max():.2f}]")
    print(f"I_gen shape: {I_gen.shape}, dtype: {I_gen.dtype}, range: [{I_gen.min():.2f}, {I_gen.max():.2f}]")
    print(f"I_bin shape: {I_bin.shape}, dtype: {I_bin.dtype}, unique vals: {torch.unique(I_bin).tolist()}")
