import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.dataset import LipSyncDataset
from src.model import SimpleLipSyncModel

def train_pipeline(audio_npy, lip_npy, epochs=500, batch_size=16):
    
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
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if epoch % 50 == 0:
            avg_loss = epoch_loss / len(loader)
            print(f"Epoch {epoch:03d} | Average MSE Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), "models/weights.pth")
    print("Training complete. Weights saved to models/weights.pth")

if __name__ == "__main__":
    train_pipeline(
        audio_npy="data/X_audio_data.npy", 
        lip_npy="data/Y_lip_data.npy"
    )