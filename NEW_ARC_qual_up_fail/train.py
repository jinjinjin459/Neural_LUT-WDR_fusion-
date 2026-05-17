import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import SeAFusionDataset
from model import SeAFusion
from loss import TotalLoss

def train():
    # 1. Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    epochs = 20
    batch_size = 16
    learning_rate = 0.001
    val_split = 0.1  # 10% for validation
    
    # Track top 3 best models: list of tuples (loss, filename)
    top_k_models = []
    max_keep = 3
    
    # 2. Dataset & DataLoader
    # 코랩에 업로드한 데이터셋 폴더 경로를 image_dir에 입력하세요.
    full_dataset = SeAFusionDataset(image_dir='/content/drive/MyDrive/your_image_folder')
    
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=2, pin_memory=True)
    
    # 3. Model & Loss & Optimizer
    model = SeAFusion().to(device)
    criterion = TotalLoss(alpha=1.0, beta=20.0).to(device) # Gradient Loss 비중 강화 + Texture Loss 추가
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 4. Training Loop
    print(f"Starting training... Train size: {train_size}, Val size: {val_size}")
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_int_loss = 0.0
        epoch_grad_loss = 0.0
        epoch_texture_loss = 0.0 # [신규] 텍스처 초기화
        
        for batch_idx, (I_org, I_gen, I_bin) in enumerate(train_loader):
            I_org = I_org.to(device)
            I_gen = I_gen.to(device)
            I_bin = I_bin.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            I_fusion, b_mask = model(I_gen, I_bin, I_bin, mode='train')
            
            # Loss (4개 리턴값 Unpacking)
            loss, l_int, l_grad, l_texture = criterion(I_fusion, I_org)
            
            # Backward & Optimize
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_int_loss += l_int.item()
            epoch_grad_loss += l_grad.item()
            epoch_texture_loss += l_texture.item() # [신규] 누적
            
        # Step scheduler at the end of the epoch
        scheduler.step()
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for I_org, I_gen, I_bin in val_loader:
                I_org = I_org.to(device)
                I_gen = I_gen.to(device)
                I_bin = I_bin.to(device)
                
                I_fusion = model(I_gen, I_bin, I_bin, mode='inference')
                v_loss, _, _, _ = criterion(I_fusion, I_org) # 리턴값 언패킹 맞춤 (4개)
                val_loss += v_loss.item()
        
        # Logging
        num_batches = len(train_loader)
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        avg_int = epoch_int_loss / num_batches if num_batches > 0 else 0
        avg_grad = epoch_grad_loss / num_batches if num_batches > 0 else 0
        avg_tex = epoch_texture_loss / num_batches if num_batches > 0 else 0 # [신규]
        
        num_val_batches = len(val_loader)
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        current_lr = optimizer.param_groups[0]['lr']
        
        # 터미널 출력에 Tex 추가
        print(f"Epoch [{epoch}/{epochs}] - "
              f"Train Loss: {avg_loss:.4f} "
              f"(Int: {avg_int:.4f}, Grad: {avg_grad:.4f}, Tex: {avg_tex:.4f}) | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"LR: {current_lr:.6f}")
              
        # Best model saving logic (Top 3) based on Val Loss
        if len(top_k_models) < max_keep or avg_val_loss < top_k_models[-1][0]:
            save_path = f"best_epoch_{epoch}_val_loss_{avg_val_loss:.4f}.pt"
            torch.save(model.state_dict(), save_path)
            top_k_models.append((avg_val_loss, save_path))
            
            # Sort ascending by loss (lowest loss is best)
            top_k_models.sort(key=lambda x: x[0])
            
            # Remove the worst model if we exceed max_keep
            if len(top_k_models) > max_keep:
                worst_loss, worst_file = top_k_models.pop(-1)
                if os.path.exists(worst_file):
                    os.remove(worst_file)
            
            print(f"--> Saved best model to '{save_path}' (Top 3 Val losses: {[round(m[0], 4) for m in top_k_models]})")
              
    print("Training Complete!")

if __name__ == '__main__':
    train()
