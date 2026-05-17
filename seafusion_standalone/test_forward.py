import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SeAFusionDataset
from model import SeAFusion
from loss import TotalLoss

def test():
    print("Testing architecture...")
    dataset = SeAFusionDataset(num_dummy=2)
    dataloader = DataLoader(dataset, batch_size=2)
    model = SeAFusion()
    criterion = TotalLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    
    for epoch in range(1, 3):
        for I_org, I_gen, I_bin in dataloader:
            print(f"I_org: {I_org.shape}, I_gen: {I_gen.shape}, I_bin: {I_bin.shape}")
            optimizer.zero_grad()
            I_fusion = model(I_gen, I_bin)
            loss, l_int, l_grad = criterion(I_fusion, I_org)
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} - Loss: {loss.item():.4f} (Int: {l_int.item():.4f}, Grad: {l_grad.item():.4f})")
        scheduler.step()
        print(f"LR after epoch {epoch}: {optimizer.param_groups[0]['lr']:.6f}")
    print("Test passed.")

if __name__ == '__main__':
    test()
