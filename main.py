import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

from src.dataset import LipSyncDataset
from src.model import SimpleLipSyncModel

def train_pipeline(audio_npy, lip_npy, epochs=500, batch_size=16):
    os.makedirs("models", exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training moving to: {device}")

    dataset = LipSyncDataset(audio_npy, lip_npy)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SimpleLipSyncModel().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Starting Training Session...")
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        loop = tqdm(loader, desc=f"Epoch {epoch:03d}/{epochs}", leave=False)
        
        for x, y in loop:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            prediction = model(x)
            prediction = prediction.transpose(1, 2)
            prediction = F.interpolate(prediction, size=y.shape[1], mode='linear', align_corners=False)
            prediction = prediction.transpose(1, 2)

            loss = criterion(prediction, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())
        
        avg_loss = epoch_loss / len(loader)
        
        print(f"Epoch {epoch:03d} Completed | Average MSE Loss: {avg_loss:.6f}")

        if epoch % 50 == 0 and epoch > 0:
            checkpoint_path = f"models/weights_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"--- Safety Checkpoint Saved: {checkpoint_path} ---")

    torch.save(model.state_dict(), "models/weights_final.pth")
    print("Training complete. Final weights saved to models/weights_final.pth")

if __name__ == "__main__":
    train_pipeline(
        audio_npy="data/X_audio_data.npy", 
        lip_npy="data/Y_lip_data.npy"
    )