import torch.nn as nn

class SimpleLipSyncModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=44):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=2, 
            batch_first=True
        )
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x[:, ::2, :] 
        out, _ = self.lstm(x)
        return self.linear(out)